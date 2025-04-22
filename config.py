# src/config.py

class Config:
    """Configuration for the BERT fine-tuning project."""
    
    # Model parameters
    model_name = "distilbert-base-uncased"  # Smaller model for better GPU utilization
    num_labels = 2  # Binary classification (positive/negative)
    
    # Training parameters
    max_seq_length = 128
    batch_size = 16  # Adjust based on GPU memory (try 8 if OOM errors occur)
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 3
    warmup_proportion = 0.1
    
    # Data parameters
    dataset_name = "imdb"
    train_split = "train"
    test_split = "test"
    validation_split = 0.1  # 10% of train data for validation
    
    # Paths
    output_dir = "outputs"
    
    # Random seed for reproducibility
    seed = 42