# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
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