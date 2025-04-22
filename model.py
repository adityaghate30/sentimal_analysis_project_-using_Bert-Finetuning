import torch
import torch.nn as nn
from transformers import AutoModel

class BertForTextClassification(nn.Module):
    def __init__(self, config):
        super(BertForTextClassification, self).__init__()
        self.num_labels = config.num_labels
        
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(config.model_name)
        
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}