from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
from nltk_utils import tokenize, lemmatize, remove_stopwords # Changed stemming to lemmatize

def find_textual_match(user_input, qst_data):
   """
   Trouve la meilleure correspondance textuelle en utilisant TF-IDF et similarité cosinus.
   Applique le même prétraitement (tokenisation, suppression des mots vides, et lemmatisation)
   aux questions de la base de connaissances et à l'entrée utilisateur pour une meilleure cohérence.
   
   Args:
       user_input (str): Question de l'utilisateur
       qst_data (dict): Données des questions/réponses au format JSON
       
   Returns:
       str: La réponse la plus pertinente ou None si aucune bonne correspondance
   """
   all_questions_processed = []
   all_responses = []
   
   # Define words to ignore during tokenization (mostly punctuation)
   ignore_punctuation = ['.', ',', ':', '?', '!', '/', '*', ';', '(', ')', '[', ']', '{', '}', '-', '_', '=', '+', '&', '%', '$', '#', '@', '~', '`', '"', "'", '<', '>', '|', '\\']

   # Preprocess all questions from the knowledge base
   for intent, contents in qst_data.items():
       for tag, content in contents.items():
           for question in content['questions']:
               # Apply the same preprocessing as in nltk_utils for consistency
               tokenized_q = tokenize(question)
               # Remove punctuation first, then stopwords, then lemmatize
               filtered_q = [word for word in tokenized_q if word not in ignore_punctuation]
               filtered_q = remove_stopwords(filtered_q) # Apply stop word removal
               lemmatized_q = [lemmatize(word.lower()) for word in filtered_q] # Changed stemming to lemmatize
               all_questions_processed.append(' '.join(lemmatized_q)) # Join back for TF-IDF
               all_responses.append(random.choice(content['responses'])) # Store a random response directly

   # Vectorisation TF-IDF
   vectorizer = TfidfVectorizer()
   tfidf_matrix = vectorizer.fit_transform(all_questions_processed)
   
   # Preprocess user input
   tokenized_user_input = tokenize(user_input)
   filtered_user_input = [word for word in tokenized_user_input if word not in ignore_punctuation]
   filtered_user_input = remove_stopwords(filtered_user_input) # Apply stop word removal
   lemmatized_user_input = [lemmatize(word.lower()) for word in filtered_user_input] # Changed stemming to lemmatize
   user_vector = vectorizer.transform([' '.join(lemmatized_user_input)])
   
   # Calcul de similarité cosinus
   similarities = cosine_similarity(user_vector, tfidf_matrix)
   best_match_idx = np.argmax(similarities)
   best_score = similarities[0, best_match_idx]

   # Seuil de similarité pour le TF-IDF.
   # Adjust this threshold based on testing.
   # A higher value means stricter matching (fewer fallbacks, but potentially more "I don't understand").
   # A lower value means looser matching (more fallbacks, but potentially less accurate ones).
   if best_score > 0.45: # Slightly increased from 0.4 for a bit more strictness. Tune as needed.
       return all_responses[best_match_idx]
   
   return None
