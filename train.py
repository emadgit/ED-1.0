from datasets import load_dataset
from transformers import AutoTokenizer


dataset = load_dataset("midalia/ED", split="train")

print("Dataset loaded successfully.")
print(f"Number of examples: {len(dataset)}")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
print("Tokenizer loaded successfully.")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")


def tokenize_function(examples):
    print("Tokenizing examples...")

    print(examples)
    # Tokenize the input text

    single_prompt = f"### Question:\n{examples['prompt'][0]}\n\n### Answer:\n{examples['response'][0]}"
    # Tokenize the single prompt
    tokenized_example = tokenizer(single_prompt, return_tensors="pt")

    token_ids = tokenized_example["input_ids"].squeeze(0).tolist()

    return {
        "input_ids": token_ids,
        "attention_mask": tokenized_example["attention_mask"][0].squeeze(0).tolist(),
    }


dataset = dataset.map(tokenize_function)
print("Dataset tokenized successfully.")
print(f"Number of tokenized examples: {len(dataset)}")
print(f"First tokenized example: {dataset[0]}")
