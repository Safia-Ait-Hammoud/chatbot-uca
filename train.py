import json
from nltk_utils import tokenize, lemmatize, bag_words, remove_stopwords # Changed stemming to lemmatize
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from modele import neural_net

# Charger les données
with open('qst.json','r', encoding="utf-8") as f:
   file = json.load(f)

# Prétraitement des données
all_words = []
tags = []
xy = []

# Define punctuation to ignore during word collection
ignore_punctuation = ['.', ',', ':', '?', '!', '/', '*', ';', '(', ')', '[', ']', '{', '}', '-', '_', '=', '+', '&', '%', '$', '#', '@', '~', '`', '"', "'", '<', '>', '|', '\\']


for intent, contents in file.items():
   for key, content in contents.items():
       tags.append(key)
       for element in content['questions']:
           qst = tokenize(element)
           # Remove punctuation first, then stopwords
           filtered_qst = [word for word in qst if word not in ignore_punctuation]
           filtered_qst = remove_stopwords(filtered_qst) # Apply stop word removal
           all_words.extend(filtered_qst)
           xy.append((qst, key)) # Keep original tokenized qst for bag_words input

# Lemmatization et nettoyage
all_words = [lemmatize(word.lower()) for word in all_words] # Changed stemming to lemmatize
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Préparation des données d'entraînement
x_train = []
y_train = []

for x, y in xy:
   # bag_words function now handles stop word removal internally
   bag = bag_words(x, all_words) 
   x_train.append(bag)
   label = tags.index(y)
   y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

# Dataset
class chat_dataset(Dataset):
   def __init__(self):
       self.n_samples = len(x_train)
       self.x_data = x_train
       self.y_data = y_train
   
   def __len__(self):
       return self.n_samples
   
   def __getitem__(self, index):
       return self.x_data[index], self.y_data[index]

# Hyperparamètres
# Experiment with these values to find the best performance for your dataset.
# hidden_size: Can be increased (e.g., 64, 128) for more complex patterns, but increases training time.
# learning_rate: Smaller values (e.g., 0.0005) can lead to more stable training but might be slower.
# n_epoch: Increase if the model is still learning, decrease if it's overfitting or early stopping triggers quickly.
# batch_size: Affects training stability and speed.
batch_size = 12
input_size = len(all_words)
hidden_size = 256 # increasing this to 64 or 128 if needed
output_size = len(tags)
learning_rate = 0.0005
n_epoch = 1500 # Can be increased, but early stopping will prevent unnecessary training
early_stopping_patience =900 # Number of epochs to wait for improvement before stopping

# Préparation des données
dataset = chat_dataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
   dataset=train_dataset,
   batch_size=batch_size,
   shuffle=True,
   num_workers=0,
   drop_last=False
)

val_loader = DataLoader(
   dataset=val_dataset,
   batch_size=batch_size,
   shuffle=False,
   num_workers=0,
   drop_last=False
)

# Modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = neural_net(input_size, hidden_size, output_size).to(device)

# Loss et optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Boucle d'entraînement
best_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(n_epoch):
   # Entraînement
   model.train()
   train_loss = 0
   for (words, labels) in train_loader:
       words = words.to(device)
       labels = labels.to(dtype=torch.long).to(device)
       
       # Forward pass
       outputs = model(words)
       loss = loss_function(outputs, labels)
       
       # Backward et optimisation
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       
       train_loss += loss.item()
   
   # Validation
   model.eval()
   val_loss = 0
   correct = 0
   total = 0
   
   with torch.no_grad():
       for (words, labels) in val_loader:
           words = words.to(device)
           labels = labels.to(dtype=torch.long).to(device)
           
           outputs = model(words)
           loss = loss_function(outputs, labels)
           val_loss += loss.item()
           
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
   
   avg_train_loss = train_loss / len(train_loader)
   avg_val_loss = val_loss / len(val_loader)
   val_accuracy = 100 * correct / total
   
   # Early stopping
   if avg_val_loss < best_loss:
       best_loss = avg_val_loss
       patience_counter = 0
       best_model_state = {
           "model_state": model.state_dict(),
           "input_size": input_size,
           "output_size": output_size,
           "hidden_size": hidden_size,
           "intents": list(file.keys()),
           "all_words": all_words,
           "tags": tags
       }
   else:
       patience_counter += 1
       if patience_counter >= early_stopping_patience:
           print(f"Early stopping triggered at epoch {epoch+1}")
           break
   
   # Affichage des statistiques
   if (epoch+1) % 100 == 0 or epoch == 0:
       print(f'Epoch [{epoch+1}/{n_epoch}], '
             f'Train Loss: {avg_train_loss:.4f}, '
             f'Val Loss: {avg_val_loss:.4f}, '
             f'Val Acc: {val_accuracy:.2f}%')

# Sauvegarde du meilleur modèle
if best_model_state:
   torch.save(best_model_state, "best_model.pth")
   print(f'Training complete. Best model saved to best_model.pth with val loss: {best_loss:.4f}')
else:
   print('Training completed without finding a better model.')
