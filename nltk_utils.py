#NLP prepocessing pipline: tokeniza --) stemming(+lower method) --) bags of words 

import nltk
import numpy as np
from nltk.tokenize import word_tokenize

import spacy # Added spacy for lemmatization

# Load the French spaCy model once
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    print("French spaCy model 'fr_core_news_sm' not found. Please run:")
    print("python -m spacy download fr_core_news_sm")
    print("Then restart your application.")
    nlp = None # Set nlp to None if model not found


'''
###Télécharge les modules nécessaires UNE SEULE FOIS### 
nltk.download('punkt')          # Tokenisation 
nltk.download('wordnet')        # Lemmatization
nltk.download('stopwords')      # Liste des stopwords
nltk.download('averaged_perceptron_tagger')  # POS tagging
nltk.download('omw-1.4')        # Support multilingue WordNet
'''


FRENCH_STOP_WORDS = set([
    "a", "à", "abord", "afin", "ah", "ai", "aie", "ainsi", "aller", "allo", "allons", "allô", "alors", "anc", "ancien", "ancienne", "après", "après-midi", "apres", "apres-midi", "approx", "approx.", "assez", "attendu", "au", "aucun", "aucune", "aujourd", "aujourd'hui", "aupres", "auquel", "aura", "aurai",
    "auraient", "aurais", "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "aut", "autre", "autres", "aux", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons", "b", "bah", "beaucoup", "bien", "big", "bon",
    "bond", "bonjour", "bonne", "boum", "bravo", "brrr", "c", "ça", "car", "ce", "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là", "cent", "cependant", "certain", "certaine", "certaines", "certains", "certes", "ces", "cet", "cette", "ceux", "ceux-ci",
    "ceux-là", "chacun", "chaque", "cher", "chère", "chères", "chers", "chez", "chiche", "chut", "chuuut", "ci", "cinq", "cinquantaine", "cinquante", "cinquantième", "cinquième", "clac", "clic", "combien", "comme", "comment", "comparable", "comparables", "compris", "concernant", "contre", "couic", "crac", "d", "da", "dans", "de", "debout",
    "dedans", "dehors", "delà", "depuis", "derriere", "derrière", "des", "desormais", "desquelles", "desquels", "dessous", "dessus", "deux", "deuxième", "deuxièmement", "devant", "devers", "devra", "devrai", "devraient", "devrais", "devrait", "devras", "devrez", "devriez", "devrions", "devrons", "devront", "diff", "différent", "différente",
    "différentes", "différents", "dire", "directe", "directement", "dit", "dite", "dits", "divers", "diverse", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "doit", "doivent", "donc", "dont", "douze", "douzième", "dring", "du", "duquel", "durant", "e", "effet", "eh", "elle", "elle-même", "elles", "elles-mêmes", "en", "encore", "enfin",
    "entre", "envers", "environ", "es", "est", "et", "etc", "etre", "eu", "eue", "eues", "euh", "eurent", "eus", "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eux-mêmes", "exactement", "excepté", "extenso", "f", "fais", "faisaient", "faisant", "fait", "faites", "façon", "feront", "fi", "flac", "floc", "font", "force", "furent",
    "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut", "g", "gens", "h", "ha", "haut", "hein", "hem", "hep", "hi", "ho", "holà", "hop", "hormis", "hors", "huit", "huitième", "hum", "hurrah", "i", "ici", "il", "ils", "importe", "j", "je", "jusqu", "jusque", "k", "l", "la", "là", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs",
    "longtemps", "lors", "lorsque", "lui", "lui-même", "m", "ma", "maint", "maintenant", "mais", "malgre", "malgré", "max", "me", "meme", "mêmes", "mes", "midi", "min", "mince", "mine", "moi", "moi-même", "moins", "mon", "mot", "moyennant", "n", "na", "ne", "neuf", "neuvième", "ni", "nommés", "non", "nonobstant", "nos", "notre", "nous", "nous-mêmes", "nul", "o",
    "oh", "ohe", "oui", "outre", "où", "p", "par", "parmi", "partant", "particulier", "particulière", "particulièrement", "pas", "passé", "pendant", "personne", "peu", "peut", "peuvent", "peux", "pff", "pfft", "pfutt", "pif", "pire", "plein", "plouf", "plus", "plusieurs", "plutôt", "possessif", "possessifs", "possible", "possibles", "pouah", "pour", "pourquoi", "pourra",
    "pourrai", "pourraient", "pourrais", "pourrait", "pourras", "pourrez", "pourriez", "pourrions", "pourrons", "pourront", "pourtant", "pouf", "pouquoi", "premier", "première", "premièrement", "pres", "presque", "presqu'", "preuve", "probante", "proches", "psitt", "puisque", "q", "qu", "quand", "quant", "quant-à-soi", "quanta", "quarante", "quatorze", "quatre", "quatre-vingt",
    "quatrième", "quatrièmement", "que", "quel", "quelconque", "quelle", "quelles", "quelqu'un", "quelque", "quelques", "quels", "qui", "quiconque", "quinze", "quoi", "quoique", "r", "rare", "rarement", "rares", "relative", "relativement", "remarquable", "remarquablement", "revoici", "revoilà", "rien", "s", "sa", "sacré", "salut", "sans", "sapristi", "sauf", "se", "seize", "selon", 
    "semblable", "semblables", "sens", "sept", "septième", "sera", "serai", "seraient", "serais", "serait", "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "seulement", "si", "sien", "sienne", "siennes", "siens", "sinon", "six", "sixième", "soi", "soi-même", "soit", "soixante", "son", "sont", "sos", "soudain", "sous", "souvent", "spec", "stop", "strictement", "subtiles", 
    "suffisant", "suffisante", "suffisantes", "suffisants", "suis", "suit", "super", "sur", "surtout", "t", "ta", "tac", "tant", "tard", "te", "tel", "telle", "tellement", "telles", "tels", "tenant", "tend", "tenir", "tente", "tes", "tic", "tien", "tienne", "tiennes", "tiens", "toc", "toi", "toi-même", "ton", "tout", "toute", "toutefois", "toutes", "tous", "treize", "trente", "tres", "trois", 
    "troisième", "troisièmement", "trop", "très", "tsoin", "tsouin", "tu", "u", "un", "une", "unes", "uniformement", "unique", "uniques", "uns", "v", "va", "vais", "vas", "vers", "via", "vif", "vifs", "vingt", "vivat", "vive", "vives", "vlan", "voici", "voilà", "vont", "vos", "votre", "vous", "vous-mêmes", "vu", "w", "x", "y", "z", "zut", "ça", "ét", "étaient", "étais", "était", "étant", "état",
    "étiez", "étions", "été", "êtes", "être", "ô","quand","je"
])

# Lemmatization using spaCy
def lemmatize(word):
    if nlp is None:
        # Fallback to lowercasing if spaCy model isn't loaded
        return word.lower()
    # Process the word with spaCy and return its lemma
    # We process a single word, so we take the first token's lemma
    doc = nlp(word)
    if doc:
        return doc[0].lemma_
    return word.lower() # Fallback if spaCy somehow fails for a word

#split sentence into array of tokens / return liste of words 
def tokenize(sentence):
    return word_tokenize(sentence)  




# Function to remove stop words
def remove_stopwords(words):
    return [word for word in words if word.lower() not in FRENCH_STOP_WORDS]


#représenter un texte sous forme numerique pour l'analyser après 
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
def bag_words(tokenized_sentence, words): 
   #tokenized_sentence= user's sentence tokenized
   # words= tous les mots des qsts dans ma BD prétraité
   
   # Apply stop word removal before lemmatization for bag of words
   filtered_sentence = remove_stopwords(tokenized_sentence)
   sentence = [lemmatize(word.lower()) for word in filtered_sentence] # Changed stemming to lemmatize
   
   #transformer words en tableau numérique (à l'aide de numpy)
   bag = np.zeros(len(words), dtype=np.float32)   #tableaux inilialisé par 0 de taille nombre des elements de words 
   # lorsque un mot de sentence existe dans words on met 1 dans sa position dans bag
   for index, element_words in enumerate(words):
       if element_words in sentence:
           bag[index] = 1
   return bag