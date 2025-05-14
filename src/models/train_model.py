import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

# Chargement des données
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

# Chargement des meilleurs paramètres
best_params = joblib.load('models/best_params.pkl')

# Entraînement du modèle avec les meilleurs paramètres
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# Sauvegarde du modèle entraîné
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/trained_model.pkl')

print("Modèle entraîné et sauvegardé avec succès.")
