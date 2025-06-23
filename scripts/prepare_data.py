# scripts/prepare_data.py

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

# Créer le dossier data/ s’il n’existe pas
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Chargement du dataset
df = pd.read_csv("data/diamonds.csv")

# Nettoyage des colonnes inutiles
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Transformation de la colonne 'table' en int
df['table'] = df['table'].astype(int)

# Encodage de la colonne 'color' par fréquence
color_freq = df['color'].value_counts(normalize=True)
df['Color'] = df['color'].map(color_freq)
df.drop(columns=['color'], inplace=True)

# Encodage de la variable cible 'cut'
le = LabelEncoder()
df['cut'] = le.fit_transform(df['cut'])

# Sauvegarde du LabelEncoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Liste des colonnes numériques
num_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']

# Traitement des outliers (IQR method)
for col in num_cols:
    Q1 = np.percentile(df[col], 25)
    Q3 = np.percentile(df[col], 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)

# Séparation X (features) et y (target)
X = df.drop(columns=["cut"]).values
y = df["cut"].values

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sauvegarde des données sous format .npy
np.save("data/x_train.npy", X_train_scaled)
np.save("data/x_test.npy", X_test_scaled)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

print("✅ Données préparées et fichiers .npy + label_encoder.pkl sauvegardés avec succès.")
