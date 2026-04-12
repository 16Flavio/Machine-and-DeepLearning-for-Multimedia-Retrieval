from flask import Flask, jsonify, send_from_directory
import os

# On initialise Flask
app = Flask(__name__, static_folder='Site_Internet', static_url_path='')

# Chemin vers tes images de voitures
DOSSIER_CARS = './data/raw/Cars'

@app.route('/')
def home():
    return app.send_static_file('index2.html')

# Route pour lister les noms de fichiers (pour l'autocomplétion)
@app.route('/api/images')
def liste_images():
    try:
        # On récupère les fichiers images
        images = [f for f in os.listdir(DOSSIER_CARS) if f.endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
        return jsonify(images)
    except FileNotFoundError:
        return jsonify({"erreur": "Dossier Cars introuvable"}), 404

# NOUVELLE ROUTE : Pour réellement envoyer le fichier image au navigateur
@app.route('/cars_dataset/<path:nom_fichier>')
def servir_car_image(nom_fichier):
    # Cette fonction va chercher l'image dans le dossier Cars et l'envoie au HTML
    return send_from_directory(DOSSIER_CARS, nom_fichier)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)