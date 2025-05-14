import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    # Charger les données en spécifiant explicitement la colonne date
    df = pd.read_csv("data/raw/raw.csv", parse_dates=['date'])  # Spécifiez la colonne 'date' pour la conversion
    df.drop(columns=['date'], inplace=True)  # Suppression de la colonne 'date'
    
    # Séparer les features (X) et la cible (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Diviser les données en jeu d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer les dossiers de sortie si nécessaire
    os.makedirs("data/processed", exist_ok=True)

    # Sauvegarder les jeux de données dans des fichiers CSV
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

if __name__ == "__main__":
    main()
