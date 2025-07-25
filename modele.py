
# définition du modèle de réseau de neurones en utilisant PyTorch 

import torch
import torch.nn as nn

class neural_net(nn.Module):
   def __init__(self,input,hidden,output): #nombre de neurones dans  : entrée , couches cachées, sortie 
       super(neural_net,self).__init__()
       self.l1=nn.Linear(input,hidden) # First hidden layer
       self.l2=nn.Linear(hidden,hidden) # Second hidden layer
       self.l3=nn.Linear(hidden,output) # Output layer
       self.dropout = nn.Dropout(0.5) # Dropout layer for regularization
       self.activation= nn.ReLU() # ReLU activation function
   def forward(self,x):   #forward pass du réseau.Elle est appelée automatiquement : output = model(input)
       out=self.l1(x)
       out=self.activation(out)
       out = self.dropout(out)  # Apply dropout after activation
       
       out=self.l2(out)
       out=self.activation(out)
       out = self.dropout(out)  # Apply dropout after activation
       
       out=self.l3(out) # Final layer, no activation or dropout after it for classification (CrossEntropyLoss expects raw logits)
       return out
