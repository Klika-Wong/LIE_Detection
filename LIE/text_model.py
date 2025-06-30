import torch
import torch.nn as nn
from transformers import BertModel

class TextModel(nn.Module):
    """
    Text model using a pretrained BERT to encode text into embeddings.
    Supports partial freezing of BERT layers for small datasets and
    adds a fully connected layer for feature compression.
    """
    def __init__(self, freeze_bert=True, bert_model_name='bert-base-uncased', freeze_layers=8, out_dim=256):
        super(TextModel, self).__init__()
        # Load pretrained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        if freeze_bert:
            # Freeze embedding layer and first `freeze_layers` encoder layers
            for name, param in self.bert.named_parameters():
                # Check if parameter belongs to encoder layers
                if name.startswith('encoder.layer'):
                    layer_num = int(name.split('.')[2])
                    if layer_num < freeze_layers:
                        param.requires_grad = False
                else:
                    # Freeze embedding, pooler, and other layers
                    param.requires_grad = False
        
        # Fully connected layer to reduce BERT output dimension
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, out_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, input_ids, attention_mask):
        """
        input_ids: tensor [batch_size, seq_length]
        attention_mask: tensor [batch_size, seq_length]
        returns: tensor [batch_size, out_dim]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # shape [batch_size, hidden_size]
        x = self.fc(pooled_output)             # shape [batch_size, out_dim]
        return x
