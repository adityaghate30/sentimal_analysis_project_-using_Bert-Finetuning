import os
import torch
import time
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from .evaluate import evaluate_model
from .utils import set_seed

def train_model(model, train_loader, val_loader, config):
    """Train the model on the given data loaders."""
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set up tensorboard for logging
    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'logs'))
    
    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Prepare learning rate scheduler
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move model to device
    model.to(device)
    
    # Track best validation accuracy
    best_val_accuracy = 0.0
    
    # Training loop
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        model.train()
        
        # Track metrics
        total_loss = 0.0
        start_time = time.time()
        
        # Training
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(progress_bar):
            # Move data to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / len(train_loader)
        training_time = time.time() - start_time
        
        # Validation
        val_results = evaluate_model(model, val_loader, device)
        val_accuracy = val_results['accuracy']
        val_loss = val_results['loss']
        
        # Log metrics
        writer.add_scalar('Training/Loss', avg_train_loss, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
        
        print(f"Training Loss: {avg_train_loss:.4f} | "
              f"Validation Loss: {val_loss:.4f} | "
              f"Validation Accuracy: {val_accuracy:.4f} | "
              f"Time: {training_time:.2f}s")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_path = os.path.join(config.output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
            }, model_path)
            print(f"Best model saved with validation accuracy: {val_accuracy:.4f}")
    
    # Save the final model
    final_model_path = os.path.join(config.output_dir, "final_model.pt")
    torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)
    
    # Close tensorboard writer
    writer.close()
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_accuracy:.4f}")
    return best_val_accuracy