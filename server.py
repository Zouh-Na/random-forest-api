import scipy.io
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Autorise les requêtes depuis Flutter

# Charger le modèle depuis un fichier .mat
mat_data = scipy.io.loadmat('random_forest_model.mat')  # Mets le bon chemin
random_forest_model = mat_data['monModele'][0, 0]  # Vérifie la structure exacte

@app.route('/')
def home():
    return "API de Maintenance Prédictive en ligne ! 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['features']
        data_array = np.array([data])
        
        prediction = random_forest_model.predict(data_array)
        
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
