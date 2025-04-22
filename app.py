# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer
import time
import os
from src.model import BertForTextClassification
from src.config import Config

@st.cache_resource
def load_model_and_tokenizer():
    """Load the model and tokenizer (cached to prevent reloading)"""
    # Load configuration
    config = Config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load model
    model = BertForTextClassification(config)
    model_path = os.path.join(config.output_dir, "best_model.pt")
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    else:
        st.error(f"Model not found at {model_path}. Please train the model first.")
    
    return model, tokenizer, device, config

def predict_sentiment(text, model, tokenizer, device, config):
    """Predict sentiment for a given text."""
    # Tokenize input
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=config.max_seq_length,
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
    
    # Get probabilities
    probs = torch.softmax(logits, dim=1)[0].tolist()
    
    # Map prediction to sentiment
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probs[prediction]
    
    return sentiment, confidence, probs

def main():
    st.set_page_config(
        page_title="BERT Sentiment Analysis",
        page_icon="😊",
        layout="wide"
    )
    
    st.title("BERT Sentiment Analysis")
    st.write("Enter text to analyze its sentiment using a fine-tuned BERT model.")
    
    # Load model and tokenizer
    with st.spinner("Loading model... (This may take a few moments)"):
        model, tokenizer, device, config = load_model_and_tokenizer()
    
    # Create text input area
    text_input = st.text_area("Enter your text:", height=150, 
                             placeholder="Type or paste text here for sentiment analysis...")
    
    # Create columns for button and examples
    col1, col2 = st.columns([1, 2])
    
    with col1:
        analyze_button = st.button("Analyze Sentiment", type="primary")
    
    with col2:
        st.write("**Try these examples:**")
        example1 = st.button("This movie was fantastic! The acting was superb and the story kept me engaged throughout.")
        example2 = st.button("Worst experience ever. Complete waste of time and money.")
        example3 = st.button("The product has some good features, but overall I was disappointed with its performance.")
    
    # Set text area based on example buttons
    if example1:
        text_input = "This movie was fantastic! The acting was superb and the story kept me engaged throughout."
        st.session_state.text_input = text_input
        analyze_button = True
    elif example2:
        text_input = "Worst experience ever. Complete waste of time and money."
        st.session_state.text_input = text_input
        analyze_button = True
    elif example3:
        text_input = "The product has some good features, but overall I was disappointed with its performance."
        st.session_state.text_input = text_input
        analyze_button = True
    
    # Process input and display results
    if analyze_button and text_input:
        with st.spinner("Analyzing sentiment..."):
            # Add slight delay for visual effect
            time.sleep(0.5)
            sentiment, confidence, probs = predict_sentiment(text_input, model, tokenizer, device, config)
        
        # Display results
        st.subheader("Results")
        
        # Create columns for sentiment and confidence
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sentiment", sentiment)
            
            # Add emoji based on sentiment
            if sentiment == "Positive":
                st.markdown("## 😊")
            else:
                st.markdown("## 😞")
        
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")
            
            # Show probability distribution
            st.write("Probability Distribution:")
            st.bar_chart({
                "Negative": probs[0],
                "Positive": probs[1]
            })
        
        # Add text analysis details
        st.subheader("Analysis Details")
        st.write(f"Text length: {len(text_input)} characters, {len(text_input.split())} words")
        
        if len(text_input) > config.max_seq_length * 4:  # rough estimate based on tokenization
            st.warning(f"⚠️ Your text is long and may have been truncated to fit the model's {config.max_seq_length} token limit.")
    
    # Add information about the model
    with st.expander("About the Model"):
        st.write(f"""
        - Model: {config.model_name}
        - Dataset: {config.dataset_name}
        - Task: Binary sentiment classification (positive/negative)
        - This model was fine-tuned on the IMDB movie reviews dataset.
        """)
        
    # Footer
    st.markdown("---")
    st.caption("BERT Sentiment Analysis - Transformer Fine-tuning Project")

if __name__ == "__main__":
    main()