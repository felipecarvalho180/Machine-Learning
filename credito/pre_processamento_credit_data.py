# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
base = pd.read_csv('credit_data.csv')
base.describe()
base.loc[base['age'] < 0]

#SOLUÇÕES PARA NUMEROS NEGATIVOS NAS COLUNAS

# apagar coluna
# base.drop('age', 1, inplace=True)

# apagar somente os registros com problemas
# base.drop(base[base.age < 0].index, inplace=True)

# preencher os valores com a média
ageMean = base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = ageMean

base.loc[pd.isnull(base['age'])]

#SOLUÇÃO PARA NUMEROS NULOS OU NAN NAS COLUNAS

previsores = base.iloc[:, 1:4].values
classes = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

#ESCALONAMENTO DE ATRIBUTOS

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
