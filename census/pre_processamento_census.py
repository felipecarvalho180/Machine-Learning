# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:51:07 2021

@author: Felipe
"""

import pandas as pd
base = pd.read_csv('census.csv')
previsores = base.iloc[:, 0:14].values
classes = base.iloc[:, 14].values

#TRANFORMANDO STRING EM NUMERICO

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_previsores= LabelEncoder()
previsores[:, 1] = labelEncoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelEncoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelEncoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelEncoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelEncoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelEncoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelEncoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelEncoder_previsores.fit_transform(previsores[:, 13])

# TRANFORMANDO CATEGORIA ORDINAL EM NOMINAL

from sklearn.compose import ColumnTransformer
oneHotEncoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
previsores = oneHotEncoder.fit_transform(previsores).toarray()

labelEncoder_classes = LabelEncoder()
classes = labelEncoder_classes.fit_transform(classes)

#ESCALONAMENTO

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)