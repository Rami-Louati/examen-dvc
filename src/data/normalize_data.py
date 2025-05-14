import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib

# Chargement des données
X_train = np.load('data/processed/X_train.npy')
X_test = np.load('data/processed/X_test.npy')

# Initialisation et entraînement du scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création des dossier
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models/data', exist_ok=True)

# Sauvegarde des datasets normalisés
np.save('data/processed/X_train_scaled.npy', X_train_scaled)
np.save('data/processed/X_test_scaled.npy', X_test_scaled)

# Sauvegarde du scaler pour réutilisation future
joblib.dump(scaler, 'models/data/scaler.pkl')
