import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(model, data_loader, device):
    """Evaluate the model on the given data loader."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")
        for input_ids, attention_mask, labels in progress_bar:
            # Move data to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Update metrics
            total_loss += loss.item()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    avg_loss = total_loss / len(data_loader)
    
    # Create results dictionary
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return results

def test_model(model, test_loader, device):
    """Test the model on the test set and print detailed metrics."""
    print("\nEvaluating model on test set...")
    
    # Evaluate on test set
    test_results = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\nTest Results:")
    print(f"Loss: {test_results['loss']:.4f}")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    print(f"F1 Score: {test_results['f1']:.4f}")
    
    return test_results