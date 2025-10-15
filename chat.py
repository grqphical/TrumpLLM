"""An application to talk with the trained TrumpLLM"""
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

base_model_name = "google/gemma-3-1b-it"
lora_model_path = "./trumpLLM"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model
model = AutoModelForCausalLM.from_pretrained(base_model_name)

chat = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0
)

running = True
print("TrumpLLM interface. Type 'exit' to quit")
while running:
    prompt = input("> ")
    if prompt.lower() == "exit":
        break

    response = chat(
        prompt,
        max_length=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
  )[0]["generated_text"]