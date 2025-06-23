import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('data/diamonds.csv')

# Exemple d'encodage de 'color' via mapping fréquence
fe = data['color'].value_counts(normalize=True)
data['color'] = data['color'].map(fe)

# Encodage label pour 'cut'
le = LabelEncoder()
data['cut'] = le.fit_transform(data['cut'])

# Séparation features et target
X = data.drop(columns=['cut']).values  # Doit être uniquement numérique ici !
y = data['cut'].values

# Sauvegarde numpy
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
np.save('data/x_train.npy', X_train)
np.save('data/x_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)
