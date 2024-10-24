import sys,os
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import pickle

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,Y_train,X_test,Y_test,models,params):
    try:
        report ={}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            greed_search = GridSearchCV(model,param,cv=3)
            greed_search.fit(X_train,Y_train)
            model.set_params(**greed_search.best_params_)
            
            model.fit(X_train,Y_train)
            Y_train_pred = model.predict(X_train)
            Y_test_pred= model.predict(X_test)
            train_model_score = r2_score(Y_train,Y_train_pred)
            test_model_score = r2_score(Y_test,Y_test_pred)
            report[list(models.keys())[i]] = test_model_score
    except Exception as e:
        raise CustomException(e,sys)
    return report

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

            
