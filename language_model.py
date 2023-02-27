####################################################
# IMPORTING PACKAGES
####################################################
#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

####################################################
# SETUP
####################################################

def activate_gpu(device_type="mac"):
    if device_type == "mac":
        if not torch.backends.mps.is_available():
            print("MPS is not available.")
            if not torch.backends.mps.is_built():
                print("MPS is not built.")
        else:
            device = torch.device("mps")
            print(f"Using device: {device}")
    if device_type == "windows":
        if not torch.backends.mps.is_available():
            print("CUDA is not available.")
            if not torch.backends.mps.is_built():
                print("CUDA is not built.")
        else:
            device = torch.device("cuda")
            print(f"Using device: {device}")

    return

####################################################
# PROCESS DATA
####################################################

def load_echr(task="binary_cls", anon=False):
    if anon == False:
        train_df = pd.read_pickle("data/echr/non-anon_train.pkl")
        val_df = pd.read_pickle("data/echr/non-anon_valid.pkl")
        test_df = pd.read_pickle("data/echr/non-anon_test.pkl")
    else:
        train_df = pd.read_pickle("data/echr/anon_train.pkl")
        val_df = pd.read_pickle("data/echr/anon_valid.pkl")
        test_df = pd.read_pickle("data/echr/anon_test.pkl")

    if task == "binary_cls":
        train_text, train_label = train_df["text"].tolist(), train_df["violated"].int().tolist()
        val_text, val_label = val_df["text"].tolist(), val_df["violated"].int().tolist()
        test_text, test_label = test_df["text"].tolist(), test_df["violated"].int().tolist()

    return train_text, train_label, val_text, val_label, test_text, test_label

def load_models(model_name="bert-based-uncased"):
    model = AutoModel.from_pretrained(model_name, return_dict=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
       
def generate_tokens(tokenizer, text, label, max_length=256):
    token = tokenizer.batch_encode_plus(
        text, 
        max_length = max_length, 
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )
    seq = torch.tensor(token['input_ids'])
    mask = torch.tensor(token['attention_mask'])
    y = torch.tensor(label).type(torch.LongTensor)

    return seq, mask, y

def create_dateloaders(seq, mask, y, sampler_type="random", batch_size=32):
    data = TensorDataset(seq, mask, y)
    if sampler_type == "random":
        sampler = RandomSampler(data)
    elif sampler_type == "sequential":
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader

####################################################
# BERT
####################################################

#%%

class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask)
      
      x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)
      
      # apply softmax activation
      x = self.softmax(x)

      return x
    
#%%

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

#%%

# push the model to GPU
model = model.to(device)

# %% optimizer

optimizer = AdamW(model.parameters(), lr=1e-3)
     
# %% class weights

class_wts = compute_class_weight('balanced', classes=np.unique(train_label), y=train_label)
#print(class_wts)

# %%

weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)
cross_entropy  = nn.NLLLoss(weight=weights) 
#cross_entropy = nn.BCEWithLogitsLoss()
epochs = 10

# %%

def train():
  
  model.train()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save model predictions
  total_preds=[]
  
  # iterate over batches
  for step,batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    # push the batch to gpu
    batch = [r.to(device) for r in batch]
 
    sent_id, mask, labels = batch

    # clear previously calculated gradients 
    model.zero_grad()        

    # get model predictions for the current batch
    preds = model(sent_id, mask)

    # compute the loss between actual and predicted values
    loss = cross_entropy(preds, labels)

    # add on to the total loss
    total_loss = total_loss + loss.item()

    # backward pass to calculate the gradients
    loss.backward()

    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # update parameters
    optimizer.step()

    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)
  
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  #returns the loss and predictions
  return avg_loss, total_preds

#%%

def evaluate():
  
  print("\nEvaluating...")
  
  # deactivate dropout layers
  model.eval()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save the model predictions
  total_preds = []

  # iterate over batches
  for step,batch in enumerate(val_dataloader):
    
    # Progress update every 50 batches.
    if step % 50 == 0 and not step == 0:
      
      # Calculate elapsed time in minutes.
      #elapsed = format_time(time.time() - t0)
            
      # Report progress.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

    # push the batch to gpu
    batch = [t.to(device) for t in batch]

    sent_id, mask, labels = batch

    # deactivate autograd
    with torch.no_grad():
      
      # model predictions
      preds = model(sent_id, mask)

      # compute the validation loss between actual and predicted values
      loss = cross_entropy(preds,labels)

      total_loss = total_loss + loss.item()

      preds = preds.detach().cpu().numpy()

      total_preds.append(preds)

  # compute the validation loss of the epoch
  avg_loss = total_loss / len(val_dataloader) 

  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds

# %% train

# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss, _ = train()
    
    #evaluate model
    valid_loss, _ = evaluate()
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')
    
# %% predict

batch_size = 32

test_data = torch.utils.data.TensorDataset(test_seq, test_mask)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

preds = []

# Set the model to evaluation mode
model.eval()

# Disable gradient computation to save memory
with torch.no_grad():

    # Iterate over the batches and make predictions
    for batch_seq, batch_mask in test_dataloader:
        # Move the batch data to the GPU if available
        batch_seq = batch_seq.to(device)
        batch_mask = batch_mask.to(device)
        
        # Make predictions for the batch
        batch_preds = model(batch_seq, batch_mask)
        batch_preds = batch_preds.detach().cpu().numpy()
        
        # Store the batch predictions
        preds.append(batch_preds)

# Concatenate the predictions for all batches
preds = np.concatenate(preds, axis=0)

# get predictions for test data
#with torch.no_grad():
  #preds = model(test_seq.to(device), test_mask.to(device))
  #preds = preds.detach().cpu().numpy()

# model's performance
preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds))

# %% load model

#load weights of best model
path = 'models/bert-0.pt'
model.load_state_dict(torch.load(path))

# %%
