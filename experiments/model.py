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
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(self.bert.pooler.dense.out_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        losses, logits = self.bert(input_ids, attention_mask, return_dict=False)
        outputs = self.dropout(logits)
        outputs = self.linear(logits)
        preds = self.sigmoid(outputs)

        return preds
    
class LegalBertMultiCls(nn.Module):
    """
    bert-small-uncased on multi-class classification
    """
    def __init__(self, legalbert, num_classes=23):
        super(LegalBertMultiCls, self).__init__()
        self.bert = legalbert 
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(self.bert.pooler.dense.out_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        losses, logits = self.bert(input_ids, attention_mask, return_dict=False)
        outputs = self.dropout(logits)
        outputs = self.linear(logits)
        preds = self.sigmoid(outputs)

        return preds
    
class LegalBertRegression(nn.Module):
    """
    bert-small-uncased on regression
    """
    def __init__(self, legalbert):
        super(LegalBertRegression, self).__init__()
        self.bert = legalbert 
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(self.bert.pooler.dense.out_features, 1)

    def forward(self, input_ids, attention_mask):
        losses, logits = self.bert(input_ids, attention_mask, return_dict=False)
        outputs = self.dropout(logits)
        outputs = self.linear(logits)

        return outputs
    
"""
functions
"""

def get_model(task, model):
    if task == "binary_cls":
        model = LegalBertBinaryCls(model)
    elif task == "multi_cls":
        model = LegalBertMultiCls(model)
    elif task == "regression":
        model = LegalBertRegression(model)

    return model