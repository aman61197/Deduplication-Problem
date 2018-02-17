# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 20:49:57 2018

@author: Aman Chaudhary
"""

#important package import 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import numpy as np

#data import 
df = pd.read_csv('Deduplication Problem - Sample Dataset.csv')


#dataframe for unique names(contain one name of each duplicate names) 
# and duplicate names(contain extra duplicate names entry)
df_unique = pd.DataFrame()
df_duplicate = pd.DataFrame()

#extraction of unique gender and dob
gender = sorted(df.gn.unique())  
date = df.dob.unique()

#iteration over gender and date for extracting of unique chunk from df
for g in gender : 

    for d in date :
        #chunk selection
        data = df[(df.gn == g)&(df.dob == d)]
        data.index = range(len(data))
        
        if len(data) == 0:  
             pass  #if entry unavailable for some gender and date combination
        else:    
             #making sinle name vector with first name and last name      
             fn_ln = (data.fn + ' ' + data.ln).str.lower()
             
             tfidf = TfidfVectorizer()
             csr_matrix = tfidf.fit_transform(fn_ln)
             words = tfidf.get_feature_names()
             
             nmf = NMF()
             nmf_features = nmf.fit_transform(csr_matrix)
             
             
             norm_features = normalize(nmf_features)
             #taking dot product to analyse identity between entries
             for i in range(len(data)) :
               current_row = norm_features[i,:]
               similarity = norm_features.dot(current_row)
               similarity[i] = 0
               
               if max(similarity) > 0.8 :
                    #appended entry in df_duplicate for similarity > 0.8 
                    df_duplicate = df_duplicate.append(data.iloc[i],ignore_index=True)
                    
                    '''
                    Making norm_features row zero for extraction of
                    last duplicate name entry into unique dataframe
                    '''
                    norm_features[i,:] = np.zeros(len(words))
               else :
                    df_unique = df_unique.append(data.iloc[i],ignore_index=True)
                    
                    
 #output                   
print(df_unique)
print(df_duplicate)

#saving dataframe to output file
df_unique.to_csv('output.csv')
  