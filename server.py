import scipy.io
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

# Charger le modèle depuis un fichier .mat
mat_data = scipy.io.loadmat('random_forest_model.mat')

# Afficher toutes les clés pour mieux comprendre la structure
print("Clés du fichier .mat:", mat_data.keys())

# Inspecter le contenu des autres clés
for key in mat_data.keys():
    print(f"Contenu de {key} : {mat_data[key]}")

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
