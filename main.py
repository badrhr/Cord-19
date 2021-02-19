# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 16:19:43 2021

@author: Administrateur
"""
############################
#######   Imports  #########
############################

import pandas as pd
from model import *
from utils import *
import pickle
import os
import warnings
warnings.filterwarnings('ignore', category=Warning)


dr = 'output'
if not os.path.exists(dr):
    os.makedirs(dr)
        
        
############################
####### Input Text #########
############################

meta = pd.read_csv("input/metadata.csv")
print(meta.shape)


#####################################
# count how many docs have abstract #
#####################################

count = 0
index = []
for i in range(len(meta)):
    #print(i)
    if type(meta.iloc[i, 8])== float:
        count += 1
    else:
        index.append(i)

print(len(index), " papers have abstract available.")

####################################
######   Extract the abstract ######
####################################
documents = meta.iloc[index, 8]
documents=documents.reset_index()
documents.drop("index", inplace = True, axis = 1)


documents = documents[:1000]

data = documents
data = data.fillna('') 
rws = data.abstract


sentences, token_lists, idx_in = preprocess(rws, samp_size=100)

methods = ['TFIDF', 'LDA', 'BERT', 'LDA_BERT']


def build(method):
    
    ntopic = 10

    # Define the topic model object

    tm = Topic_Model(k = ntopic, method = method)
    # Fit the topic model by chosen method
    tm.fit(sentences, token_lists)
    
    
    if method != 'LDA_BERT':
        # Evaluate using metrics
        with open("input/{}.file".format(tm.id), "wb") as f:
            pickle.dump(tm, f, pickle.HIGHEST_PROTOCOL)

    print('Coherence:', get_coherence(tm, token_lists, 'c_v'))
    print('Silhouette Score:', get_silhouette(tm))

    # visualize and save img
    visualize(tm)
   

   
for method in methods:
    print("*********************************** Results for", str(method), "************************")
    build(method)
    print("\n")
    
    
