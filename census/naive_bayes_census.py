#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 19:35:31 2021

@author: felipe
"""

import pandas as pd

base = pd.read_csv('census.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
                
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEncoder_previsores= LabelEncoder()
previsores[:, 1] = labelEncoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelEncoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelEncoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelEncoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelEncoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelEncoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelEncoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelEncoder_previsores.fit_transform(previsores[:, 13])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)

# Teste de previsores
resultado = classificador.predict(previsores_teste)

# Comparativo de resposta para indentificar erros
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, resultado)
matriz = confusion_matrix(classe_teste, resultado)