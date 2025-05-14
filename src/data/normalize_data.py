import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd


# Charger les données avec pandas
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')

# Appliquer la normalisation avec StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création des dossier
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models/data', exist_ok=True)

# Sauvegarder les données normalisées
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('data/processed/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('data/processed/X_test_scaled.csv', index=False)

# Sauvegarde du scaler pour réutilisation future
joblib.dump(scaler, 'models/data/scaler.pkl')
