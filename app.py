from flask import Flask, jsonify, request, send_from_directory
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from part1_unimodal.search import (
    search as run_search,
    search_uploaded as run_search_uploaded,
    projection_3d as run_projection_3d,
)
from part2_multimodal.faiss_index import search_text as mm_search_text, search_image as mm_search_image

app = Flask(__name__, static_folder='Site_Internet', static_url_path='')

DOSSIER_CARS = './data/raw/Cars'
DOSSIER_FLICKR = './data/raw/Flickr8k_dataset/Images'

@app.route('/')
def home():
    return app.send_static_file('index2.html')

@app.route('/api/images')
def liste_images():
    try:
        images = [f for f in os.listdir(DOSSIER_CARS) if f.endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
        return jsonify(images)
    except FileNotFoundError:
        return jsonify({"erreur": "Dossier Cars introuvable"}), 404

@app.route('/cars_dataset/<path:nom_fichier>')
def servir_car_image(nom_fichier):
    return send_from_directory(DOSSIER_CARS, nom_fichier)


@app.route('/flickr_dataset/<path:nom_fichier>')
def servir_flickr_image(nom_fichier):
    return send_from_directory(DOSSIER_FLICKR, nom_fichier)

@app.route('/api/search_multimodal', methods=['POST'])
def api_search_multimodal():
    try:
        top_k = int(request.form.get('top_k') or (request.get_json(silent=True) or {}).get('top_k') or 10)
    except (TypeError, ValueError):
        return jsonify({"erreur": "top_k invalide"}), 400

    try:
        if 'image' in request.files and request.files['image'].filename:
            from PIL import Image as _PILImage
            uploaded = request.files['image']
            img = _PILImage.open(uploaded.stream).convert('RGB')
            payload = mm_search_image(img, top_k, uploaded_name=uploaded.filename)
        else:
            data = request.get_json(silent=True) or request.form
            text = (data.get('text') or '').strip()
            if not text:
                return jsonify({"erreur": "Fournissez un texte ou une image."}), 400
            payload = mm_search_text(text, top_k)
        return jsonify(payload)
    except ValueError as e:
        return jsonify({"erreur": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"erreur": f"Index CLIP manquant: {e}"}), 500
    except Exception as e:
        return jsonify({"erreur": f"Erreur serveur: {e}"}), 500

@app.route('/api/projection_3d', methods=['GET'])
def api_projection_3d():
    descriptor = (request.args.get('descriptor') or '').strip()
    if not descriptor:
        return jsonify({"erreur": "Paramètre 'descriptor' requis."}), 400
    try:
        return jsonify(run_projection_3d(descriptor))
    except ValueError as e:
        return jsonify({"erreur": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"erreur": f"Fichier de descripteurs manquant: {e}"}), 500
    except Exception as e:
        return jsonify({"erreur": f"Erreur serveur: {e}"}), 500


@app.route('/api/search', methods=['POST'])
def api_search():
    try:
        if 'image' in request.files and request.files['image'].filename:
            from PIL import Image as _PILImage
            img = _PILImage.open(request.files['image'].stream).convert('RGB')
            descriptors = request.form.getlist('descriptors')
            metric = request.form.get('metric', '')
            try:
                top_k = int(request.form.get('top_k') or 10)
            except (TypeError, ValueError):
                return jsonify({"erreur": "top_k invalide"}), 400
            query_label = (request.form.get('query_label') or '').strip() or None
            payload = run_search_uploaded(
                image=img,
                descriptors=descriptors,
                metric=metric,
                top_k=top_k,
                query_label=query_label,
            )
        else:
            data = request.get_json(silent=True) or {}
            payload = run_search(
                filename=data.get('filename', ''),
                descriptors=data.get('descriptors') or data.get('descriptor'),
                metric=data.get('metric', ''),
                top_k=data.get('top_k', 10),
            )
        return jsonify(payload)
    except ValueError as e:
        return jsonify({"erreur": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"erreur": f"Fichier de descripteurs manquant: {e}"}), 500
    except Exception as e:
        return jsonify({"erreur": f"Erreur serveur: {e}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)