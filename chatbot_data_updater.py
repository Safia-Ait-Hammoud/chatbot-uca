#enrichitle fichier "qst.json" existant avec de nouvelles données à partir d’un fichier Excel "chatbot_nouvelles_donnees.xlsx"
#If the existing data file is modified, the training script must be re-run
import json
import pandas as pd
import shutil
import datetime

# Nom du fichier JSON principal
json_file = "qst.json"

# Créer une copie de sauvegarde du fichier JSON existant avec un timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup_file = f"qst_backup_{timestamp}.json"
shutil.copyfile(json_file, backup_file)
print(f"Copie de sauvegarde créée : {backup_file}")

# Charger le fichier JSON existant
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Charger les nouvelles données depuis le fichier Excel
df = pd.read_excel("chatbot_nouvelles_donnees.xlsx")

# Parcourir chaque ligne du fichier Excel
for _, row in df.iterrows():
    cat = row['Catégorie']
    subcat = row['Sous-catégorie']
    typ = row['Type']
    contenu = row['Contenu']

    # Initialiser la structure de catégorie si elle n'existe pas
    if cat not in data:
        data[cat] = {}

    # Initialiser la sous-catégorie si elle n'existe pas
    if subcat not in data[cat]:
        data[cat][subcat] = {"questions": [], "responses": []}

    # Ajouter le contenu à la bonne liste (questions ou responses) s'il n'est pas déjà présent
    if typ == "question" and contenu not in data[cat][subcat]["questions"]:
        data[cat][subcat]["questions"].append(contenu)
    elif typ == "response" and contenu not in data[cat][subcat]["responses"]:
        data[cat][subcat]["responses"].append(contenu)

# Réécrire les données mises à jour dans le fichier JSON principal
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Le fichier JSON a bien été modifié et sauvegardé.")
