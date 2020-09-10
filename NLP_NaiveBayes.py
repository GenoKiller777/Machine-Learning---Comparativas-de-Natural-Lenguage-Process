# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:47:36 2020

@author: darwi
"""

# Natural Language Processing - Procesamiento de Lenguaje Natural -Naive Bayes

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

#En la clasificacion se usa Naive Bayes 

# División del conjunto de datos en el conjunto de entrenamiento y Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Entrenando al modelo Naive Bayes en el set de entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicción de los resultados del conjunto de pruebas
y_pred = classifier.predict(X_test)

# Haciendo la Matriz de Confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
