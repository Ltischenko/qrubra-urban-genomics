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

Y = df.Latitude

x = df.drop(labels=['Samples','Latitude','Longitude','Batch','Samples_ID','Population_code'], axis=1)

if scaling == 1:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    treatment = scaler.fit(x)
    X = scaler.transform(x)
else:
    X = x
    pass

del(x)

X2 = X.sample(n=int(SNPsize),axis='columns')

seed = random.randint(0,1000)
# seed = 289 # if fixed
x_train, x_test, y_train_lat, y_test_lat = train_test_split(X2, Y, test_size = 80, random_state=int(seed))

y_train_lat.plot.kde()
y_test_lat.plot.kde()
plt.savefig('distrib_lat'+str(seed)+'.png')
plt.close()

metrics = pd.DataFrame()
metrics['metrics']= ['model','N_SNPs','lat_mae','lat_rmse','lat_best_parameters','long_mae','long_rmse','long_best_parameters','mean_dist_km','std_dist_km']

from sklearn.metrics import r2_score



if model == 'Linear':
    from sklearn.linear_model import LinearRegression
    reg1 = LinearRegression()
    reg1.fit(x_train,y_train_lat)

    Y_lat_pred = reg1.predict(x_test)
    plt.scatter(y_test_lat,Y_lat_pred)
    plt.savefig('scatter_lat'+str(seed)+'.png')

    r2_score(y_test_lat,Y_lat_pred)

    mae  = mean_absolute_error(y_test_lat,Y_lat_pred)
    mse = mean_squared_error(y_test_lat,Y_lat_pred)
    rmse = np.sqrt(mse)

    errors = [model,SNPsize,mae,rmse,'Linear']
    #metrics['Linear_lat'] = errors
    metrics

else:
    pass


if model == 'RF':
    # # RandomForestRegressor

    from sklearn.ensemble import RandomForestRegressor
    mod2 = RandomForestRegressor()

    hyperparameters2 = {'min_samples_leaf':[1,2,5], 
                        'n_estimators':[1000,2000,5000], 
                      'min_samples_split':[2,5,10]}

    #hyperparameters2 = {'min_samples_leaf':[2], 
    #                    'n_estimators':[5],
    #                    'min_samples_split':[2]}

    reg2 = GridSearchCV(mod2, hyperparameters2, scoring='neg_mean_squared_error', n_jobs=Ncpus, verbose=0)
    reg2.fit(x_train,y_train_lat)

    Y_lat_pred = reg2.predict(x_test)
    plt.scatter(y_test_lat,Y_lat_pred)
    plt.savefig('scatter_lat'+str(seed)+'.png')

    r2_score(y_test_lat,Y_lat_pred)

    mae  = mean_absolute_error(y_test_lat,Y_lat_pred)
    mse = mean_squared_error(y_test_lat,Y_lat_pred)
    rmse = np.sqrt(mse)

    errors = [model,SNPsize,mae,rmse,reg2.best_params_]
    #metrics['RF_lat'] = errors

    metrics

else:
    pass


if model == 'GBoost':
    # # GradientBoosting

    from sklearn.ensemble import GradientBoostingRegressor
    mod3 = GradientBoostingRegressor()
    
    hyperparameters3 = {'learning_rate':[0.01,0.1,1,10,100],'n_estimators':[10,50,100,200,400,500,1000]}
    
    '''
    hyperparameters3 = {'learning_rate':0.1, 'n_estimators':5}
    '''
    
    reg3 = GridSearchCV(mod3, hyperparameters3, cv=5, scoring='neg_mean_squared_error', n_jobs=Ncpus, verbose=0)
    
    reg3.fit(x_train,y_train_lat)
    Y_lat_pred = reg3.predict(x_test)

    plt.scatter(y_test_lat,Y_lat_pred)
    plt.savefig('scatter_lat'+str(seed)+'.png')


    r2_score(y_test_lat,Y_lat_pred)
    mae  = mean_absolute_error(y_test_lat,Y_lat_pred)
    mse = mean_squared_error(y_test_lat,Y_lat_pred)
    rmse = np.sqrt(mse)

    errors = [model,SNPsize,mae,rmse,reg3.best_params_]
    #metrics['GBoost_lat'] = errors

else:
    pass


if model == 'KNN':

    # # K-Nearest-Neighbors

    from sklearn.neighbors import KNeighborsRegressor
    mod6 = KNeighborsRegressor()

    hyperparameters6 = {'n_neighbors':[1,3,5,10,15],
                        'weights':['uniform','distance'],
                        'algorithm':['auto','ball_tree','kd_tree','brute'],
                        'leaf_size':[1,10,30,100],
                        'p':[1,2]
                       }
    '''
    
    hyperparameters6 = {'n_neighbors':[1],
                        'weights':['uniform'],
                        'algorithm':['auto'],
                        'leaf_size':[1],
                        'p':[1]
                       }
    '''

    reg6 = GridSearchCV(mod6, hyperparameters6, scoring='neg_mean_squared_error', n_jobs=Ncpus, verbose=0)

    reg6.fit(x_train, y_train_lat)

    Y_lat_pred = reg6.predict(x_test)


    plt.scatter(y_test_lat,Y_lat_pred)
    plt.savefig('scatter_lat'+str(seed)+'.png')

    mae  = mean_absolute_error(y_test_lat,Y_lat_pred)
    mse = mean_squared_error(y_test_lat,Y_lat_pred)
    rmse = np.sqrt(mse)

    errors = [model,SNPsize,mae,rmse,reg6.best_params_]
    #metrics['KNN_lat'] = errors
    metrics

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

Y = df.Longitude


x_train, x_test, y_train_long, y_test_long = train_test_split(X, Y, test_size = 80, random_state=int(seed))

y_train_long.plot.kde()
y_test_long.plot.kde()
plt.savefig('distrib_long'+str(seed)+'.png')
plt.close()


if model == 'Linear':
    from sklearn.linear_model import LinearRegression
    reg1 = LinearRegression()

    reg1.fit(x_train,y_train_long)

    Y_long_pred = reg1.predict(x_test)
    plt.scatter(y_test_long,Y_long_pred)
    plt.savefig('scatter_long'+str(seed)+'.png')


    from sklearn.metrics import r2_score
    r2_score(y_test_long,Y_long_pred)

    mae  = mean_absolute_error(y_test_long,Y_long_pred)
    mse = mean_squared_error(y_test_long,Y_long_pred)
    rmse = np.sqrt(mse)

    errors.extend([mae,rmse,'Linear'])
#    metrics['results'] = errors
    metrics

else:
    pass



if model == 'RF':
    # # RandomForestRegressor

    from sklearn.ensemble import RandomForestRegressor
    mod2 = RandomForestRegressor()

    hyperparameters2 = {'min_samples_leaf':[1,2,5], 
                        'n_estimators':[1000,2000,5000], 
                      'min_samples_split':[2,5,10]}

    '''
    
    hyperparameters2 = {'min_samples_leaf':[2], 
                        'n_estimators':[5],
                        'min_samples_split':[2]}
    '''

    reg2 = GridSearchCV(mod2, hyperparameters2, scoring='neg_mean_squared_error', n_jobs=Ncpus, verbose=0)

    reg2.fit(x_train,y_train_long)

    Y_long_pred = reg2.predict(x_test)
    plt.scatter(y_test_long,Y_long_pred)
    plt.savefig('scatter_long'+str(seed)+'.png')

    r2_score(y_test_long,Y_long_pred)

    mae  = mean_absolute_error(y_test_long,Y_long_pred)
    mse = mean_squared_error(y_test_long,Y_long_pred)
    rmse = np.sqrt(mse)

    errors.extend([mae,rmse,reg2.best_params_])
#    metrics['results'] = errors

    metrics

else:
    pass


if model == 'GBoost':
    # # GradientBoosting

    from sklearn.ensemble import GradientBoostingRegressor
    mod3 = GradientBoostingRegressor()
    hyperparameters3 = {'learning_rate':[0.001,0.1,1],
                        'n_estimators':[1000,2000,5000]}
    '''
    hyperparameters3 = {'learning_rate':[0.1],'n_estimators':[5]}
    '''
    
    reg3 = GridSearchCV(mod3, hyperparameters3, scoring='neg_mean_squared_error', n_jobs=Ncpus, verbose=0)
    reg3.fit(x_train,y_train_long)

    Y_long_pred = reg3.predict(x_test)

    plt.scatter(y_test_long,Y_long_pred)
    plt.savefig('scatter_long'+str(seed)+'.png')

    r2_score(y_test_long,Y_long_pred)
    mae  = mean_absolute_error(y_test_long,Y_long_pred)
    mse = mean_squared_error(y_test_long,Y_long_pred)
    rmse = np.sqrt(mse)

    errors.extend([mae,rmse,reg3.best_params_])
#    metrics['results'] = errors

else:
    pass




if model == 'KNN':

    # # K-Nearest-Neighbors

    from sklearn.neighbors import KNeighborsRegressor
    mod6 = KNeighborsRegressor()

    hyperparameters6 = {'n_neighbors':[1,3,5,10,15],
                        'weights':['uniform','distance'],
                        'algorithm':['auto','ball_tree','kd_tree','brute'],
                        'leaf_size':[1,10,30,100],
                        'p':[1,2]
                       }
    '''

    hyperparameters6 = {'n_neighbors':[1],
                        'weights':['uniform'],
                        'algorithm':['auto'],
                        'leaf_size':[1],
                        'p':[1]
                       }
    '''

    reg6 = GridSearchCV(mod6, hyperparameters6, scoring='neg_mean_squared_error', n_jobs=Ncpus, verbose=0)
    reg6.fit(x_train, y_train_long)

    Y_long_pred = reg6.predict(x_test)

    plt.scatter(y_test_long,Y_long_pred)
    plt.savefig('scatter_long'+str(seed)+'.png')
    plt.close()

    mae  = mean_absolute_error(y_test_long,Y_long_pred)
    mse = mean_squared_error(y_test_long,Y_long_pred)
    rmse = np.sqrt(mse)

    errors.extend([mae,rmse,reg6.best_params_])
#    metrics['results'] = errors

else:
    pass

plt.close()


# # Calculation of geographic error in km.
forMap = pd.DataFrame()
forMap['lat'] = y_test_lat
forMap['long'] = y_test_long
forMap['pred_lat'] = Y_lat_pred
forMap['pred_long'] = Y_long_pred

print(forMap)

# function to calculate the km
def calc_erreur_km(lat,long,plat,plong):
    xdis = abs(plong-long)*111.32*np.cos(math.radians(lat))
    #print(xdis)
    ydis = abs(plat-lat)*110.574
    #print(ydis)
    dist = np.hypot(xdis,ydis)
    return dist

# apply calculus
forMap['error_dist_km'] = forMap.apply(lambda x: calc_erreur_km(x.lat,x.long,x.pred_lat,x.pred_long),axis=1)

print(forMap)

# get an error distribution

forMap.error_dist_km.plot.kde()
#y_train_long.plot.kde()
plt.savefig('distrib_error_km_'+str(seed)+'.png')
plt.close()

# calculate mean and std for the error
mean = np.mean(forMap['error_dist_km'])
std = np.std(forMap['error_dist_km'])

# append calculus to output table
#dic1 = {'metrics':'mean_error_km',model:str(mean)}
#dic2 = {'metrics':'std_error_km',model:str(std)}
#errorM = pd.DataFrame(dic1, index=[0])
#errorSTD = pd.DataFrame(dic2, index=[0])
#metrics2 = pd.concat([metrics,errorM],ignore_index=True)
#metrics3 = pd.concat([metrics2,errorSTD], ignore_index=True)

errors.extend([mean,std])

metrics['results'] = errors

metrics.to_csv('metrics'+str(model)+'N'+str(SNPsize)+'_seed'+str(seed)+'.csv', index=False)
#plt.close()
