import sqlite3

conn = sqlite3.connect('uca_form.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nom TEXT NOT NULL,
    prenom TEXT NOT NULL,
    etablissement TEXT NOT NULL,
    email TEXT NOT NULL,
    question TEXT NOT NULL
)
''')

conn.commit()
conn.close()
