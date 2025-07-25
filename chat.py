import json
import torch
import random
from modele import neural_net
from nltk_utils import bag_words, tokenize
from FIND_textual_match import find_textual_match

with open('qst.json','r', encoding="utf-8") as f :
   qst=json.load(f)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model and its parameters
# IMPORTANT: You need to run `train.py` first to generate `best_model.pth`
try:
    data=torch.load("best_model.pth")
    model_state=data["model_state"]
    input_size=data["input_size"]
    output_size=data["output_size"]
    hidden_size=data["hidden_size"]
    intents=data["intents"]
    all_words=data["all_words"]
    tags=data["tags"]

    # Create and load the trained weights
    model=neural_net(input_size,hidden_size,output_size)
    model.load_state_dict(model_state)
    model.eval() # Set the model to evaluation mode
except FileNotFoundError:
    print("Error: best_model.pth not found. Please run train.py first.")
    # Fallback or exit strategy if model is not found
    model = None # Indicate that the model is not loaded

def traitement(sentence):
    if model is None:
        return "Le modèle de chatbot n'est pas encore entraîné ou chargé. Veuillez contacter l'administrateur."

    sentence_tokenized = tokenize(sentence)
    x = bag_words(sentence_tokenized, all_words)     # x is numpy array of dim 1
    x = x.reshape(1, x.shape[0])     # transform it to 2D (batch_size=1,input_size=number of elements) (model input)
    x = torch.from_numpy(x).to(device)  # convert it to a pytorch tensor

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Find section:
    section = None
    for intent_name, contents in qst.items():
        if tag in contents:
            section = intent_name
            break

    # Confidence-based response strategy
    # Adjust these thresholds based on your testing to balance accuracy and fallback behavior.
    if prob.item() < 0.50:  # Low confidence threshold: Fallback to textual match or generic response
        best_match = find_textual_match(sentence, qst)
        if best_match:
            return best_match
        return "Je n'ai pas bien compris votre question. Pouvez-vous la reformuler ou essayer avec d'autres mots ?"
    elif prob.item() < 0.65 : # Intermediate confidence zone: Ask for clarification
        # Increased threshold from 0.60 to 0.70 to make the model more confident before giving a direct answer.
        if section =='general':
            content = qst[section][tag]
            response = random.choice(content['responses'])
            return response
        return f"Je pense que vous parlez de '{tag}'. Pouvez-vous préciser votre question ?"
    else: # High confidence: Provide direct answer
        if section and tag in qst[section]:
            content = qst[section][tag]
            response = random.choice(content['responses'])
            return response
        else:
            # This case should ideally not be hit if the model is well-trained and qst.json is consistent.
            return "Je n'ai pas trouvé de réponse spécifique pour cette question, même avec une bonne confiance."
