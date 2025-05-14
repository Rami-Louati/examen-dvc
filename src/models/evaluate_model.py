import numpy as np
import os
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Chargement des données
X_test = np.load('data/processed/X_test_scaled.npy')
y_test = np.load('data/processed/y_test.npy')

# Chargement du modèle entraîné
model = joblib.load('models/trained_model.pkl')

# Prédictions
predictions = model.predict(X_test)

# Calcul des métriques
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

# Sauvegarde des métriques
metrics = {
    'mse': mse,
    'r2': r2,
    'mae': mae
}
os.makedirs('metrics', exist_ok=True)
with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Sauvegarde des prédictions
os.makedirs('data', exist_ok=True)
np.save('data/predictions.npy', predictions)

print("Évaluation terminée. Scores et prédictions sauvegardés.")
