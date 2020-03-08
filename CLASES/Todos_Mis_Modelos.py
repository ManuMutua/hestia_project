# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 20:05:11 2020

@author: Nadal
"""


#Esta clase pretende hacer una predicción usando todos los modelos disponibles
#esto puede ayudarnos focalizar el análisis predictivo.

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



class Todos_Mis_Modelos:
    '''
    Vamos a estimar todos los modelos
    '''
    
    def __init__(self,train_X, train_Y, test_X ,test_Y ):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        
    #Estimaremos varios tipos de modelos:
    def Muchos_Modelos(self):
        
        #Modelos de regresion:

        mi_regresion = linear_model.LinearRegression()
        mi_regresion_logistica = LogisticRegression(random_state=0, solver='lbfgs')
        reg_bayes = linear_model.BayesianRidge()
        mi_regresion_svm = svm.SVR()
        mi_regresion_knn = KNeighborsRegressor(n_neighbors=5)
        mi_regresion_gauss = GaussianNB()
        mi_regresion_tree = tree.DecisionTreeRegressor(max_depth=2)
        mi_regresion_rf =  RandomForestRegressor(max_depth=2, random_state=0, min_samples_split=2, bootstrap=True)
        
        #Estimación del modelo:
        mi_regresion.fit( self.test_X , self.test_Y )
        mi_regresion_logistica.fit( self.test_X , self.test_Y )
        reg_bayes.fit( self.test_X , self.test_Y )
        mi_regresion_svm.fit(self.test_X , self.test_Y )
        mi_regresion_knn.fit( self.test_X , self.test_Y )
        mi_regresion_gauss.fit( self.test_X , self.test_Y )
        mi_regresion_tree.fit( self.test_X , self.test_Y )
        mi_regresion_rf.fit( self.test_X , self.test_Y )
        
        #realizamos las predicciónes con el conjunto de test:
        mi_prediccion = mi_regresion.predict(self.test_X)
        mi_prediccion_log = mi_regresion_logistica.predict(self.test_X)
        mi_prediccion_bayes = reg_bayes.predict(self.test_X)
        mi_prediccion_svm = mi_regresion_svm.predict(self.test_X)
        mi_prediccion_knn = mi_regresion_knn.predict(self.test_X)
        mi_prediccion_gaus = mi_regresion_gauss.predict(self.test_X)
        mi_prediccion_tree = mi_regresion_tree.predict(self.test_X)
        mi_prediccion_rf = mi_regresion_rf.predict(self.test_X)
        #mi criterio de error:
        rmse_regresion = mean_squared_error(self.test_Y, mi_prediccion)
        rmse_regresion_logistica = mean_squared_error(self.test_Y, mi_prediccion_log)
        rmse_regresion_b = mean_squared_error(self.test_Y, mi_prediccion_bayes)
        rmse_regresion_svm = mean_squared_error(self.test_Y, mi_prediccion_svm)
        rmse_regresion_knn = mean_squared_error(self.test_Y, mi_prediccion_knn)
        rmse_regresion_gauss = mean_squared_error(self.test_Y, mi_prediccion_gaus)
        rmse_regresion_tree = mean_squared_error(self.test_Y, mi_prediccion_tree)
        rmse_regresion_rf = mean_squared_error(self.test_Y, mi_prediccion_rf)
        
        #creamos un conjunto de datos que tenga todos nuestras medidas de error:
        
        rmse_dicc = {
                "Regresión_Lineal" : rmse_regresion
                ,"Regresión_logística" : rmse_regresion_logistica
                ,"Bayes" : rmse_regresion_b
                ,"SVM" : rmse_regresion_svm
                ,"KNN" : rmse_regresion_knn
                ,"Gauss" : rmse_regresion_gauss
                ,"Árbol_decisión" : rmse_regresion_tree
                ,"Bosque_aleatorio" : rmse_regresion_rf}
        
        
        return rmse_dicc
    
    
    
        

        
    

    