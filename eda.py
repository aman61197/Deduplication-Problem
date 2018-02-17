# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 04:39:00 2018

@author: Aman Chaudhary
"""

#data import and exploration
import pandas as pd

df = pd.read_csv('Deduplication Problem - Sample Dataset.csv')

#EDA for basic insight of data
print(df.head())

print(df.info())

print(df.describe())

unique_ln = list(df.ln.unique())
print(unique_ln)



unique_fn = list(df.fn.unique())
print(unique_fn)

#Using multi-indexing, we can have better insight of our data.
df = df.set_index(['gn','dob','fn'])
df = df.sort_index()
print(df)