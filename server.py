import scipy.io
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

# Charger le modèle depuis un fichier .mat
mat_data = scipy.io.loadmat('random_forest_model.mat')

# Vérifier les clés du fichier pour voir si 'monModele' est présent
print("Clés du fichier .mat:", mat_data.keys())

# Vérifier la structure exacte de l'objet
if 'monModele' in mat_data:
    random_forest_model = mat_data['monModele'][0, 0]
    print("Modèle chargé avec succès.")
else:
    print("'monModele' n'est pas présent dans le fichier .mat.")

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
