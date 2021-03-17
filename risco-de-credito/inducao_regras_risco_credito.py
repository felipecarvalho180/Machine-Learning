#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 01:09:06 2021

@author: felipe
"""

import Orange

base = Orange.data.Table('risco_credito.csv')
base.domain

cs2_learner = Orange.classification.rules.CN2Learner()
classificador = cs2_learner(base)
for regras in classificador.rule_list:
    print(regras)
    
# TESTE
    
resultado = classificador((['boa', 'alta', 'nenhuma', 'acima_35'], ['ruim', 'alta', 'adequada', '0_15']))
for i in resultado:
    # Para pegar o valor de acordo com o valor da classe
    print(base.domain.class_var.values[i])