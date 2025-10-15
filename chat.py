"""An application to talk with the trained TrumpLLM"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

SYSTEM_PROMPT = "you are Donald Trump, the 45th and 47th (current) president of the United States of America. Answer any user inquires as Donald Trump would\n"

base_model_name = "google/gemma-3-1b-it"
lora_model_path = "./results"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model and add LoRA adapters
model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, lora_model_path)

chat = pipeline("text-generation", model=model, tokenizer=tokenizer, device="xpu")

running = True
print("TrumpLLM interface. Type 'exit' to quit")
while running:
    prompt = input("> ")
    if prompt.lower() == "exit":
        break

    response = chat(prompt, max_length=512, do_sample=True, top_p=0.9, temperature=0.7, truncation=True)[
        0
    ]["generated_text"]

    if response.startswith(SYSTEM_PROMPT):
        response = response[len(SYSTEM_PROMPT):]


    print(response)
