import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Charger le CSV
df = pd.read_csv('data/diamonds.csv')
  # Assure-toi que ce fichier est bien dans data/

# Exemple : la colonne cible s'appelle 'target', adapte selon ton CSV
X = df.drop(columns=['cut']).values  # Les features
y = df['cut'].values  # La cible

# Séparer train/test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sauvegarder au format .npy
np.save('data/x_train.npy', X_train)
np.save('data/x_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

print("Fichiers .npy créés avec succès dans le dossier data/")
