from datasets import load_dataset
from transformers import AutoTokenizer


dataset = load_dataset("midalia/ED", split="train")

print("Dataset loaded successfully.")
print(f"Number of examples: {len(dataset)}")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
print("Tokenizer loaded successfully.")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")


tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"Tokenizer pad token: {tokenizer.pad_token}")
print(f"Tokenizer pad token id: {tokenizer.pad_token_id}")
print(f"Tokenizer padding side: {tokenizer.padding_side}")


def tokenize_function(batch):
    print("Tokenizing batch...")

    print(batch)
    # Tokenize the input text

    prompts = [
        f"###Question:\n {p}\n###Answer:\n {a}"
        for p, a in zip(batch["prompt"], batch["response"])
    ]

    # Tokenize the prompts
    tokenize_new_rows = tokenizer(
        prompts,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    token_ids = tokenize_new_rows["input_ids"]
    attention_mask = tokenize_new_rows["attention_mask"]

    return {
        "input_ids": token_ids,
        "attention_mask": attention_mask,
        "labels": token_ids,
    }


dataset = dataset.map(tokenize_function, batched=True)
print("Dataset tokenized successfully.")
print(f"Number of tokenized examples: {len(dataset)}")
print(f"First tokenized example: {dataset[0]}")
