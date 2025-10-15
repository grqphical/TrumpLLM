import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

posts = pd.read_json("posts.json", lines=True)
posts = posts[["content"]]

pd.set_option('display.max_colwidth', None)
print(posts.iloc[4])

def clean_content(row):
    prefix = "<p>"
    suffix = "</p>"
    content = row["content"]
    if content.startswith(prefix):
        content = content[len(prefix):]
    if content.endswith(suffix):
        content = content[:-len(suffix)]
    return content

posts["content"] = posts.apply(clean_content, axis=1)

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(posts)
dataset = dataset.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

def tokenize(batch):
    """Converts the text into tokens for the LLM to train"""
    tokens = tokenizer(batch["content"], truncation=True, padding="max_length", max_length=1024)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, batched=True)

# ==========Training==========
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
model.config.use_cache = False  # Needed for training
model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    learning_rate=5e-5,
    save_total_limit=2,
    fp16=True,
    report_to=[]
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

# Train
trainer.train()

# Save
model.save_pretrained("./trumpLLM")
tokenizer.save_pretrained("./trumpLLM")