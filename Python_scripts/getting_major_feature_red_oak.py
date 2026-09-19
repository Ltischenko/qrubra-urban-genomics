#!/usr/bin/env python
# coding: utf-8


import sys, fileinput

import pandas as pd
import math
import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold

from sklearn import manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import check_random_state
from sklearn.inspection import permutation_importance


# make sure all arguments are there
try:
    file = sys.argv[1]
    var = sys.argv[2]
#    ncol = int(sys.argv[3])
except:
    print('three arguments are expected: file, variable to predict')
    sys.exit(1)


# read file
df = pd.read_csv(file)

print(file)
print('file read')

# x with SNPs only
x = df.drop(labels=['Samples','Latitude','Longitude','Batch','Samples_ID','Population_code'], axis=1)

print(x)


# Define algo and hyper-parameters
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

#from sklearn.ensemble import GradientBoostingRegressor
#reg=GradientBoostingRegressor(learning_rate=0.1, n_etimators=500)



# Train model
reg.fit(x,df[var])

# Prepare scores object
scores= pd.DataFrame(columns=['feature','importance'])
print(scores)

#get sample to test
x_train, x_test, y_train, y_test = train_test_split(x, df[var], test_size = 200, random_state=2314)

#make prediction with test
pred_y = reg.predict(x_test)


# Calculate Performance with all features
P2 = np.sqrt(np.mean((y_test-pred_y) ** 2))

print('model performance: ')
print(P2)

# Function to calculate score for a single column
for col in x.columns:
    nx = x_test.copy()
    values = x_test[col].values.copy()
    np.random.shuffle(values)
    nx[col] = values
    pred_y = reg.predict(nx)
    P1 = np.sqrt(np.mean((y_test- pred_y) ** 2))
    score = P1-P2
    scores.loc[len(scores)] = [col, score]

print(scores)

# Save scores
scores.to_csv(file[:-4]+'_scores_'+var+'.csv', index=False)

