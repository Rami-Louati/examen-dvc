import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Chargement des données
X_train = np.load('data/processed/X_train_scaled.npy')
y_train = np.load('data/processed/y_train.npy')

# Définition du modèle et des hyperparamètres à tester
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Lancement du GridSearch
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Sauvegarde des meilleurs paramètres
os.makedirs('models', exist_ok=True)
joblib.dump(grid_search.best_params_, 'models/best_params.pkl')

print("Best parameters found:", grid_search.best_params_)
