import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Chargement dataset
data = pd.read_csv('data/diamonds.csv')

# Supposons que 'color' est une colonne catégorielle avec des valeurs comme 'F'
# Encodage de 'color' en valeurs numériques (label encoding)
le_color = LabelEncoder()
data['color'] = le_color.fit_transform(data['color'])

# Si 'cut' est aussi catégorielle (target), tu peux aussi la transformer (mais en gardant pour target)
le_cut = LabelEncoder()
data['cut'] = le_cut.fit_transform(data['cut'])

# Drop une colonne inutile si besoin
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)

# Séparation features/target
X = data.drop('cut', axis=1).values  # toutes colonnes sauf target
y = data['cut'].values

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# Sauvegarde numpy
np.save('data/x_train.npy', X_train)
np.save('data/y_train.npy', y_train)
np.save('data/x_test.npy', X_test)
np.save('data/y_test.npy', y_test)

print("Fichiers .npy créés avec succès")
