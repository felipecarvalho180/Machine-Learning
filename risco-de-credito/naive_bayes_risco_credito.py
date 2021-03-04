#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 18:48:24 2021

@author: felipe
"""

import pandas as pd

base = pd.read_csv('risco_credito.csv')
previsores = base.iloc[:, 0: 4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
label_encoder_previsores = LabelEncoder()
previsores[:, 0] = label_encoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = label_encoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 2] = label_encoder_previsores.fit_transform(previsores[:, 2])
previsores[:, 3] = label_encoder_previsores.fit_transform(previsores[:, 3])

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)

#Teste com histórico boa, divida alta, garantia nenhuma, renda > 35
resultado = classificador.predict([[0,0,1,2], [3,0,0,0]])
print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)