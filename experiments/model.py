"""
import packages
"""

import torch.nn as nn

"""
classes
"""

class LegalBertBinaryCls(nn.Module):
    """
    bert-small-uncased on binary classification
    """
    def __init__(self, legalbert):
        super(LegalBertBinaryCls, self).__init__()
        self.bert = legalbert 
        self.linear = nn.Linear(self.bert.pooler.dense.out_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        losses, logits = self.bert(input_ids, attention_mask, return_dict=False)
        outputs = self.linear(logits)
        preds = self.sigmoid(outputs)
        # squeeze
        preds = preds.squeeze(1)

        return preds
    
class LegalBertMultiCls(nn.Module):
    """
    bert-small-uncased on multi-class classification
    """
    def __init__(self, legalbert, num_classes):
        super(LegalBertMultiCls, self).__init__()
        self.bert = legalbert 
        self.linear = nn.Linear(self.bert.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask, return_dict=False)
        outputs = self.linear(outputs)
        preds = self.softmax(outputs)

        return preds
    
class LegalBertRegression(nn.Module):
    """
    bert-small-uncased on regression
    """
    def __init__(self, legalbert):
        super(LegalBertRegression, self).__init__()
        self.bert = legalbert 
        self.linear = nn.Linear(self.bert.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask, return_dict=False)
        outputs = self.linear(outputs)

        return outputs