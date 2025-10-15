import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

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