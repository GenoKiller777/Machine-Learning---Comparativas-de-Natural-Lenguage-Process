# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 20:42:31 2020

@author: darwi
"""

# Natural Language Processing - Procesamiento de Lenguaje Natural - KNN

# Importacion de Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importacion del DataSet
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Limpieza de texto
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word)for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Crear el Bag of Words (Bolsa de palabras) Tokenizacion - Matriz Dispersa
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Dividir el DataSet en conjuntos de entrenamiento y conjunto de testing.
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,
                                  metric = "minkowski",
                                  p = 2)
classifier.fit(X_train, Y_train)
                                  

# Predicción de los Resultados con el Conjunto de Testing 
y_pred = classifier.predict(X_test)

# Elaborar una matriz de Confusión 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)