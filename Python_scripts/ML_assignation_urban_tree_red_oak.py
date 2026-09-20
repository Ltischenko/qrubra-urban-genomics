#!/usr/bin/env python
# coding: utf-8

# # This is the script to run machine learning with a subset
# # of SNPs.
# # The random SNP selection is done without replacement so
# # that a SNP can not be picked up twice.
# # 


#### libraries import
import sys, fileinput

import pandas as pd
import math
import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold

from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import check_random_state


########


# ## first get the data


try:
    infile = sys.argv[1]
    scaling = sys.argv[2]
    model = sys.argv[3]
    SNPsize = int(sys.argv[4])
    Ncpus = int(sys.argv[5])
    print('Input file: ', infile)
    print('scaling: ', scaling)
    print('model: ', model)
    print('SNP number: ', SNPsize)
    print('CPUs: ',Ncpus)
    
except:
    print(''' there's some missing arguments
        Usage: 
            training3.py infile scaling model SNP_size Ncpus
            infile
            scaling: 1 = yes, 0 = no
            model = Linear, RF, GBoost, or KNN
            SNPsize = number of SNPs
            Ncpus: number of cores to use
            ''')
    sys.exit(1)


    
df = pd.read_csv(infile)

# replace missing values with reference genotype, tests have been done using column mean.
for col in df.columns[1:-5]:
    df[col] = df[col].astype(float)
    df[col] = df[col].fillna(df[col].mean())

df1 = df[df['Population_code'] != 'VILLE']
df2 = df[df['Population_code'] == 'VILLE']

Y = df1.Latitude

x = df1.drop(labels=['Samples','Latitude','Longitude','Batch','Samples_ID','Population_code'], axis=1)
xv = df2.drop(labels=['Samples','Latitude','Longitude','Batch','Samples_ID','Population_code'], axis=1)


if scaling == 1:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    treatment = scaler.fit(x)
    X = scaler.transform(x)
    #XV = scaler.transform(xv)
else:
    X = x
    pass

del(x)

X2 = X


metrics = pd.DataFrame()
metrics['metrics']= ['model','N_SNPs','lat_mae','lat_rmse','lat_best_parameters','long_mae','long_rmse','long_best_parameters','mean_dist_km','std_dist_km']

from sklearn.metrics import r2_score



if model == 'Linear':
    from sklearn.linear_model import LinearRegression
    reg1 = LinearRegression()
    reg1.fit(X2,Y)

    Y_lat_pred = reg1.predict(X2)
    plt.scatter(Y,Y_lat_pred)
    plt.savefig('scatter_lat'+'.png')

    r2_score(Y,Y_lat_pred)

    mae  = mean_absolute_error(Y,Y_lat_pred)
    mse = mean_squared_error(Y,Y_lat_pred)
    rmse = np.sqrt(mse)

    errors = [model,SNPsize,mae,rmse,'Linear']
    #print(metrics)
    
    Y_ville_lat = reg1.predict(xv) #XV si propre
    
    out = df2['Samples']
    out['latitude'] = Y_ville_lat.tolist()

else:
    pass


if model == 'RF':
    # # RandomForestRegressor

    from sklearn.ensemble import RandomForestRegressor
    mod2 = RandomForestRegressor()

    hyperparameters2 = {'min_samples_leaf':[5], 
                        'n_estimators':[5000], 
                      'min_samples_split':[5]}


    reg2 = GridSearchCV(mod2, hyperparameters2, scoring='neg_mean_squared_error', n_jobs=Ncpus, verbose=0)
    reg2.fit(X2,Y)

    Y_lat_pred = reg2.predict(X2)
    plt.scatter(Y,Y_lat_pred)
    plt.savefig('scatter_lat'+'.png')

    r2_score(Y,Y_lat_pred)

    mae  = mean_absolute_error(Y,Y_lat_pred)
    mse = mean_squared_error(Y,Y_lat_pred)
    rmse = np.sqrt(mse)

    errors = [model,SNPsize,mae,rmse,reg2.best_params_]
    #metrics['RF_lat'] = errors

    metrics
    Y_ville_lat = reg2.predict(xv)

    out = df2['Samples']
    out['latitude'] = Y_ville_lat.tolist()

else:
    pass


if model == 'GBoost':
    # # GradientBoosting

    from sklearn.ensemble import GradientBoostingRegressor
    mod3 = GradientBoostingRegressor()
    
    hyperparameters3 = {'learning_rate':[0.01],'n_estimators':[400]}
    
    reg3 = GridSearchCV(mod3, hyperparameters3, cv=5, scoring='neg_mean_squared_error', n_jobs=Ncpus, verbose=0)
    
    reg3.fit(X2,Y)
    Y_lat_pred = reg3.predict(X2)

    plt.scatter(Y,Y_lat_pred)
    plt.savefig('scatter_lat'+'.png')


    r2_score(Y,Y_lat_pred)
    mae  = mean_absolute_error(Y,Y_lat_pred)
    mse = mean_squared_error(Y,Y_lat_pred)
    rmse = np.sqrt(mse)

    errors = [model,SNPsize,mae,rmse,reg3.best_params_]
    #metrics['GBoost_lat'] = errors
    
    Y_ville_lat = reg3.predict(xv)

    out = df2['Samples']
    out['latitude'] = Y_ville_lat.tolist()


else:
    pass


if model == 'KNN':

    # # K-Nearest-Neighbors

    from sklearn.neighbors import KNeighborsRegressor
    mod6 = KNeighborsRegressor()

    hyperparameters6 = {'n_neighbors':[15],
                        'weights':['uniform'],
                        'algorithm':['auto'],
                        'leaf_size':[100],
                        'p':[1]
                       }

    reg6 = GridSearchCV(mod6, hyperparameters6, scoring='neg_mean_squared_error', n_jobs=Ncpus, verbose=0)

    reg6.fit(X2, Y)

    Y_lat_pred = reg6.predict(X2)


    plt.scatter(Y,Y_lat_pred)
    plt.savefig('scatter_lat'+'.png')

    mae  = mean_absolute_error(Y,Y_lat_pred)
    mse = mean_squared_error(Y,Y_lat_pred)
    rmse = np.sqrt(mse)

    errors = [model,SNPsize,mae,rmse,reg6.best_params_]
    #metrics['KNN_lat'] = errors
    metrics
    
    Y_ville_lat = reg6.predict(xv)

    out = df2['Samples']
    out['latitude'] = Y_ville_lat.tolist()


else:
    pass

plt.close()




#########################################################################################
#
#
#
# LONGITUDE
#
#
#
#
########################################################################################

Y2 = df1.Longitude


if model == 'Linear':
    from sklearn.linear_model import LinearRegression
    reg21 = LinearRegression()

    reg21.fit(X2,Y2)

    Y_long_pred = reg21.predict(X2)
    plt.scatter(Y2,Y_long_pred)
    plt.savefig('scatter_long'+'.png')


    from sklearn.metrics import r2_score
    r2_score(Y2,Y_long_pred)

    mae  = mean_absolute_error(Y2,Y_long_pred)
    mse = mean_squared_error(Y2,Y_long_pred)
    rmse = np.sqrt(mse)

    errors.extend([mae,rmse,'Linear'])
#    metrics['results'] = errors
    metrics
    
    Y_ville_long = reg21.predict(xv)

    out['longitude'] = Y_ville_long.tolist()


else:
    pass



if model == 'RF':
    # # RandomForestRegressor

    from sklearn.ensemble import RandomForestRegressor
    mod2 = RandomForestRegressor()

    hyperparameters2 = {'min_samples_leaf':[5], 
                        'n_estimators':[1000], 
                      'min_samples_split':[2]}

    reg22 = GridSearchCV(mod2, hyperparameters2, scoring='neg_mean_squared_error', n_jobs=Ncpus, verbose=0)

    reg22.fit(X2,Y2)

    Y_long_pred = reg22.predict(X2)
    plt.scatter(Y2,Y_long_pred)
    plt.savefig('scatter_long'+'.png')

    r2_score(Y2,Y_long_pred)

    mae  = mean_absolute_error(Y2,Y_long_pred)
    mse = mean_squared_error(Y2,Y_long_pred)
    rmse = np.sqrt(mse)

    errors.extend([mae,rmse,reg22.best_params_])

    metrics
    Y_ville_long = reg22.predict(xv)

    out['longitude'] = Y_ville_long.tolist()


else:
    pass


if model == 'GBoost':
    # # GradientBoosting

    from sklearn.ensemble import GradientBoostingRegressor
    mod3 = GradientBoostingRegressor()
    hyperparameters3 = {'learning_rate':[0.001],
                        'n_estimators':[2000]}
    
    reg23 = GridSearchCV(mod3, hyperparameters3, scoring='neg_mean_squared_error', n_jobs=Ncpus, verbose=0)
    reg23.fit(X2,Y2)

    Y_long_pred = reg23.predict(X2)

    plt.scatter(Y2,Y_long_pred)
    plt.savefig('scatter_long'+'.png')

    r2_score(Y2,Y_long_pred)
    mae  = mean_absolute_error(Y2,Y_long_pred)
    mse = mean_squared_error(Y2,Y_long_pred)
    rmse = np.sqrt(mse)

    errors.extend([mae,rmse,reg23.best_params_])
    
    Y_ville_long = reg23.predict(xv)

    out['longitude'] = Y_ville_long.tolist()


else:
    pass




if model == 'KNN':

    # # K-Nearest-Neighbors

    from sklearn.neighbors import KNeighborsRegressor
    mod6 = KNeighborsRegressor()

    hyperparameters6 = {'n_neighbors':[15],
                        'weights':['uniform'],
                        'algorithm':['ball_tree'],
                        'leaf_size':[100],
                        'p':[1]
                       }

    reg26 = GridSearchCV(mod6, hyperparameters6, scoring='neg_mean_squared_error', n_jobs=Ncpus, verbose=0)
    reg26.fit(X2, Y2)

    Y_long_pred = reg26.predict(X2)

    plt.scatter(Y2,Y_long_pred)
    plt.savefig('scatter_long'+'.png')
    plt.close()

    mae  = mean_absolute_error(Y2,Y_long_pred)
    mse = mean_squared_error(Y2,Y_long_pred)
    rmse = np.sqrt(mse)

    errors.extend([mae,rmse,reg26.best_params_])
    
    Y_ville_long = reg26.predict(xv)

    out['longitude'] = Y_ville_long.tolist()


else:
    pass

print(metrics)

# Create output DataFrame safely
out = df2[['Samples']].copy()
out['latitude'] = Y_ville_lat.tolist()  # Make sure this variable matches your latitude prediction
out['longitude'] = Y_ville_long.tolist()  # Likewise for longitude prediction

# Save to CSV
out.to_csv(f'prediction_{model}_N{SNPsize}.csv', index=False)



#out.to_csv(f'prediction_{model}_N{SNPsize}.csv', index=False)



