import os
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch

# Step 1: Preprocess HTML Files
def extract_text_from_html(html_folder):
    """Extract text from all HTML files in the folder"""
    texts = []
    for filename in os.listdir(html_folder):
        if filename.endswith(".html"):
            filepath = os.path.join(html_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, "html.parser")
                text = soup.get_text()
                texts.append(text)
    return texts

# Step 2: Tokenization using Salesforce CodeGen tokenizer
def tokenize_texts(texts, tokenizer):
    """Tokenize a list of texts"""
    return tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

# Step 3: Create Dataset for Training
def create_dataset(tokenized_texts):
    """Create a HuggingFace Dataset from tokenized inputs"""
    input_ids = tokenized_texts['input_ids']
    attention_mask = tokenized_texts['attention_mask']
    dataset = Dataset.from_dict({
        'input_ids': input_ids,
        'attention_mask': attention_mask
    })
    return dataset

# Step 4: Model training
def train_model(dataset, model_name="Salesforce/codegen-350M-multi"):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_steps=10_000,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        logging_dir='./logs',  # directory for storing logs
        logging_steps=200,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Training the model
    trainer.train()

def main():
    html_folder = "./html_data"  # Path to your HTML files folder

    # Step 1: Extract text from HTML
    texts = extract_text_from_html(html_folder)
    print(f"Extracted {len(texts)} documents.")

    # Step 2: Load tokenizer and tokenize texts
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
    tokenized_texts = tokenize_texts(texts, tokenizer)

    # Step 3: Create dataset
    dataset = create_dataset(tokenized_texts)

    # Step 4: Train the model
    train_model(dataset)

if __name__ == "__main__":
    main()
