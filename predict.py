# predict.py

import torch
import argparse
from transformers import AutoTokenizer
from src.model import BertForTextClassification
from src.config import Config

def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for a given text."""
    # Tokenize input
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        prediction = torch.argmax(logits, dim=1).item()
    
    # Map prediction to sentiment
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = torch.softmax(logits, dim=1)[0][prediction].item()
    
    return sentiment, confidence

def main():
    parser = argparse.ArgumentParser(description='Predict sentiment of a text')
    parser.add_argument('--text', type=str, required=True, help='Text to predict sentiment for')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load model
    model = BertForTextClassification(config)
    checkpoint = torch.load(f"{config.output_dir}/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Make prediction
    sentiment, confidence = predict_sentiment(args.text, model, tokenizer, device)
    
    # Print result
    print(f"\nText: {args.text}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()