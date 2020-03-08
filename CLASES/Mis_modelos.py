# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 21:02:15 2020

@author: Nadal
"""
#En esta clase detallaremos todos los modelos por separado:

#librerías que usaremos:
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression



class Mis_modelos_sencillos:
    '''
    Empezamos a modelar
    
    '''
    def __init__(self,train_X, train_Y, test_X ,test_Y ):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
    
    
    #regresión lineal:
    def lm(self):
        
        #creamos la regresión:
        mi_regresión = linear_model.LinearRegression()
        
        #Estimación del modelo:
        mi_regresión.fit( self.test_X , self.test_Y )
        
        #realizamos las predicciónes con el conjunto de test:
        mi_predicción = mi_regresión.predict(self.test_X)
        
        #mi criterio de error:
        rmse_regresión = mean_squared_error(self.test_Y, mi_predicción)
        
        print('El error de la regresión lineal es: '+str(rmse_regresión))
        
        return rmse_regresión
    
    #regresión logística:
    def lr(self):
        
        mi_regresion = LogisticRegression(random_state=0, solver='lbfgs')
        
        mi_regresion.fit(self.test_X , self.test_Y)

        mi_predicción=mi_regresion.predict(self.test_X)
        
        #mi criterio de error:
        rmse_regresión = mean_squared_error(self.test_Y, mi_predicción)
        
        print('El error de la regresión logística es: '+str(rmse_regresión))
        
        return rmse_regresión
        
    #regresión bayesiana:
    def br( self ):
        
        mi_regresion = linear_model.BayesianRidge()
        
        mi_regresion.fit(self.test_X , self.test_Y)

        mi_predicción=mi_regresion.predict(self.test_X)
        
        #mi criterio de error:
        rmse_regresión = mean_squared_error(self.test_Y, mi_predicción)
        
        print('El error de la regresión bayesiana es: '+str(rmse_regresión))
        
        return rmse_regresión
        
    #maquina de soporte vectorial:
    def svm( self ):
        mi_regresion = svm.SVR()
        
        mi_regresion.fit(self.test_X , self.test_Y)

        mi_predicción=mi_regresion.predict(self.test_X)
        
        #mi criterio de error:
        rmse_regresión = mean_squared_error(self.test_Y, mi_predicción)
        
        print('El error de SVM  es: '+str(rmse_regresión))
        
        return rmse_regresión
        
    #KNN
    def knn(self, n_neighbors ):
        
        mi_regresion = KNeighborsRegressor(n_neighbors)
        
        
        mi_regresion.fit(self.test_X , self.test_Y)

        mi_predicción=mi_regresion.predict(self.test_X)
        
        #mi criterio de error:
        rmse_regresión = mean_squared_error(self.test_Y, mi_predicción)
        
        print('Números de vecinos : '+ str(n_neighbors))
        print('El RMSE de KNN es: '+str(rmse_regresión))
        
        return rmse_regresión
        
    #naive bayes
    def nb( self ):
        
        mi_regresion = GaussianNB()
        
        mi_regresion.fit(self.test_X , self.test_Y)

        mi_predicción=mi_regresion.predict(self.test_X)
        
        #mi criterio de error:
        rmse_regresión = mean_squared_error(self.test_Y, mi_predicción)
        
        print('El error de la regresión de Gauss es: '+str(rmse_regresión))
        
        return rmse_regresión
        
    #arboles de decisión:
    def tree(self, max_depth ):

        mi_regresion = tree.DecisionTreeRegressor(max_depth=max_depth)
        
        mi_regresion.fit(self.test_X , self.test_Y)

        mi_predicción=mi_regresion.predict(self.test_X)
        
        #mi criterio de error:
        rmse_regresión = mean_squared_error(self.test_Y, mi_predicción)
        
        print('Max_depth es :'+str(max_depth))
        print('El error del árbol es: '+str(rmse_regresión))
        
        return str(rmse_regresión)
        
    #bosque aleatorio:
    def rf(self, bootstrap, max_depth):
        
        mi_regresion =  RandomForestRegressor(max_depth = max_depth, random_state=0, min_samples_split=2, bootstrap = bootstrap)
        
        mi_regresion.fit(self.test_X , self.test_Y)

        mi_predicción=mi_regresion.predict(self.test_X)
        
        #mi criterio de error:
        rmse_regresión = mean_squared_error(self.test_Y, mi_predicción)
        
        print('Aplicando bootstrap: '+str(bootstrap))
        print('El RMSE de RF es: '+str(rmse_regresión))
        
        return rmse_regresión
        
   
    
    
    






        



