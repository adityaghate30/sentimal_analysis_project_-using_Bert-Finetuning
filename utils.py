import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_training_history(train_losses, val_losses, val_accuracies, output_dir):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def load_best_model(model, config):
    """Load the best model checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(config.output_dir, "best_model.pt")
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1} "
              f"with validation accuracy: {checkpoint['val_accuracy']:.4f}")
    else:
        print("No saved model found. Using the current model state.")
    
    model.to(device)
    return model, device