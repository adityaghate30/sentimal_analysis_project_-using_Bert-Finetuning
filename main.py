import os
import torch
import time
from src.config import Config
from src.data_loader import DataProcessor
from src.model import BertForTextClassification
from src.train import train_model
from src.evaluate import test_model
from src.utils import load_best_model

def main():
    """Main function to run the BERT fine-tuning process."""
    start_time = time.time()
    
    # Load configuration
    config = Config()
    print("\n===== BERT Fine-tuning for Text Classification =====")
    print(f"Dataset: {config.dataset_name}")
    print(f"Model: {config.model_name}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max sequence length: {config.max_seq_length}")
    print("====================================================\n")
    
    # Load and process data
    print("Loading and processing data...")
    data_processor = DataProcessor(config)
    train_loader, val_loader, test_loader = data_processor.load_and_process_data()
    
    # Initialize model
    print("\nInitializing model...")
    model = BertForTextClassification(config)
    
    # Train model
    print("\nStarting training...")
    best_val_accuracy = train_model(model, train_loader, val_loader, config)
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    model, device = load_best_model(model, config)
    
    # Test model
    test_results = test_model(model, test_loader, device)
    
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n===== Training Summary =====")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test accuracy: {test_results['accuracy']:.4f}")
    print(f"Test F1 score: {test_results['f1']:.4f}")
    print("===========================")

if __name__ == "__main__":
    main()