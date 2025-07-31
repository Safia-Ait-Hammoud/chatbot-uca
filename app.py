from flask import Flask, render_template,request,session, jsonify,redirect
import sqlite3
from chat import traitement as traitement_chatbot

app=Flask(__name__)
app.secret_key='123@321'
@app.route('/',methods=["GET","POST"])
def traitement():
    if 'history' not in session:
        session['history'] = []
    return render_template('accueil.html', history=session['history'])

@app.route('/api_repondre', methods=["POST"])
def api_repondre():
    data = request.get_json()
    question = data.get('question')
    reponse = traitement_chatbot(question)

    # Historique en session
    session['history'].append({'question': question, 'reponse': reponse})
    session.modified = True

    return jsonify({'reponse': reponse})

@app.route('/')
def accueil():
    return render_template('accueil.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/formulaire')
def formulaire():
    return render_template('formulaire.html')

@app.route('/submit', methods=['POST'])
def submit():
    nom = request.form['nom']
    prenom = request.form['prenom']
    etablissement = request.form['etablissement']
    email = request.form['email']
    question = request.form['question']
    conn = sqlite3.connect('uca_form.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO questions (nom, prenom, etablissement, email, question)
        VALUES (?, ?, ?, ?, ?)
    ''', (nom, prenom, etablissement, email, question))
    conn.commit()
    conn.close()
    return redirect('/confirmation')

@app.route('/confirmation')
def confirmation():
    return render_template('confirmation.html')

if __name__ == '__main__':
    app.run()
