import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
    def load_and_process_data(self):
        """Load dataset and prepare it for training."""
        print(f"Loading {self.config.dataset_name} dataset...")
        
        # Load dataset
        dataset = load_dataset(self.config.dataset_name)
        
        # Split training data into train and validation
        train_texts = dataset[self.config.train_split]['text']
        train_labels = dataset[self.config.train_split]['label']
        
        # Create train and validation splits
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts,
            train_labels,
            test_size=self.config.validation_split,
            random_state=self.config.seed
        )
        
        # Process the data
        train_encodings = self._tokenize_data(train_texts)
        val_encodings = self._tokenize_data(val_texts)
        test_encodings = self._tokenize_data(dataset[self.config.test_split]['text'])
        
        # Create PyTorch datasets
        train_dataset = self._create_tensor_dataset(
            train_encodings, 
            torch.tensor(train_labels)
        )
        
        val_dataset = self._create_tensor_dataset(
            val_encodings, 
            torch.tensor(val_labels)
        )
        
        test_dataset = self._create_tensor_dataset(
            test_encodings, 
            torch.tensor(dataset[self.config.test_split]['label'])
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        print(f"Data preparation complete. Train samples: {len(train_dataset)}, "
              f"Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def _tokenize_data(self, texts):
        """Tokenize the input texts."""
        return self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors='pt'
        )
    
    def _create_tensor_dataset(self, encodings, labels):
        """Create a PyTorch TensorDataset from encodings and labels."""
        return TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            labels
        )