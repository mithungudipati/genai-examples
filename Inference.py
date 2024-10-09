import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_dir):
    """Load the fine-tuned model and tokenizer from the directory."""
    # Load the fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    
    # Load the corresponding tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=100, temperature=0.7):
    """Generate text based on the input prompt."""
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text with the model
    output = model.generate(
        **inputs, 
        max_length=max_length,         # Maximum length of the generated text
        temperature=temperature,       # Controls randomness: lower values for less randomness
        num_return_sequences=1,        # Return one generated sequence
        no_repeat_ngram_size=2,        # Prevent repetition
        early_stopping=True            # Stop when the model seems done
    )
    
    # Decode the generated tokens back into text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    # Path to the directory where your model is saved (replace with your path)
    model_dir = "./saved_model"

    # Load the trained model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_dir)
    
    # Move the model to the appropriate device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("Model loaded successfully!")

    while True:
        # Get the input prompt from the user
        prompt = input("\nEnter your prompt (or type 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        
        # Generate text based on the prompt
        generated_text = generate_text(prompt, model, tokenizer)
        
        # Print the generated text
        print("\nGenerated Text:")
        print(generated_text)

if __name__ == "__main__":
    main()
