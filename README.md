# Chatbot Université Cadi Ayyad (UCA)

Ce projet est une application web de type **chatbot intelligent** développée pour répondre automatiquement aux questions fréquentes des étudiants de l'Université Cadi Ayyad (UCA). Il utilise une interface conviviale, un backend Flask, et un modèle NLP entraîné pour comprendre et répondre aux requêtes étudiantes.

## Fonctionnalités

- Interface de chat utilisateur intuitive

- Prise en charge des reformulations via NLP (traitement du langage naturel)

- Modèle de réseau de neurones entraîné pour comprendre et classer les intentions

- Système d'entraînement du modèle basé sur des données FAQ personnalisables

- Interface pour soumettre des questions non résolues

- Base de données SQLite intégrée pour stocker les formulaires utilisateurs

- Mise à jour des données via fichiers Excel (.xlsx)

## Structure du projet
```
├── app.py                      # Point d'entrée Flask
├── chat.py                     # Logique de traitement de questions
├── train.py                     # Script d'entraînement du modèle
├── best_model.pth             # Modèle NLP entraîné
├── qst.json                   # Données des questions/réponses
├── chatbot_data_updater.py     # Script pour mettre à jour les données du chatbot
├── chatbot_nouvelles_donnees.xlsx # Nouvelles données pour mise à jour
├── FIND_textual_match.py # Matching flou des textes
├── nltk_utils.py              # Fonctions NLP
├── modele.py                   # Définition du modèle
├── db.py                      # Connexion base de données
├── uca_form.db                # Base SQLite des formulaires
├── static/
│   ├── images/                # Images du chatbot
│   │   ├── blanc.png
│   │   ├── blanc_inv.png
│   │   ├── bot.png
│   │   ├── marron.png
│   │   ├── presidence.jpg
│   │   ├── profile-user.png
│   │   └── send.png
│   └── style.css             # Styles CSS
└── templates/                 # Templates HTML
    ├── accueil.html          # Home page
    ├── about.html            # About page
    ├── formulaire.html       # Contact form
    └── confirmation.html     # Form confirmation
    └── header.html           # Header partial
```

## Installation et exécution

### 1. Prerequisites

- Python 3.7+
- pip (gestionnaire de paquets)

### 2. Installation des dépendances
```bash
pip install flask torch nltk pandas numpy openpyxl scikit-learn python-Levenshtein spacy
python -m spacy download fr_core_news_sm
