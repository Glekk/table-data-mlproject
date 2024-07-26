import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve


RANDOM_STATE=42


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('models', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Splitting train and test data')
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
            'RandomForestClassifier': RandomForestClassifier(random_state=RANDOM_STATE),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'XGBClassifier': XGBClassifier(random_state=RANDOM_STATE),
            'LGBMClassifier': LGBMClassifier(random_state=RANDOM_STATE, verbosity=0)
            }

            param_grid = {
            'RandomForestClassifier':{
                'n_estimators': [64,100,200],
                'max_depth': [4,8,16,32,None],
                'max_features': ['sqrt', 'log2'],
                'criterion': ['gini', 'entropy']
                },
            'GradientBoostingClassifier':{
                'n_estimators': [16,32,64,100],
                'learning_rate': [0.005, 0.01, 0.05, 0.1],
                'subsample': [0.7,0.85,1],
                'max_depth': [8,16,32,64],
                'max_features': ['auto', 'sqrt', 'log2']
                },
            'XGBClassifier':{
                'learning_rate': [0.005, 0.01, 0.05, 0.1],
                'n_estimators': [16,32,64,100],
                'max_depth': [8,16,32,64],
                'subsample': [0.7,0.85,1],
                'colsample_bytree': [0.5, 0.65, 1],
                'colsample_bylevel': [0.5, 0.65, 1],
                'colsample_bynode': [0.5, 0.65, 1],
                'reg_alpha': [0,1,1.2],
                'reg_lambda': [0,1,1.2,1.4],
                'scale_pos_weight': [0.5,1,1.5]
                },
            'LGBMClassifier':{
                'learning_rate': [0.005, 0.01, 0.05, 0.1],
                'n_estimators': [16,32,64,100],
                'num_leaves': [8,12,16,31],
                'boosting_type' : ['gbdt', 'dart'],
                'objective' : ['binary'],
                'colsample_bytree' : [0.5, 0.65, 1],
                'subsample' : [0.7,0.85,1],
                'reg_alpha' : [0,1,1.2],
                'reg_lambda' : [0,1,1.2,1.4],
                'scale_pos_weight' : [0.5,1,1.5]
                }
            }

            model_results = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                           models=models, param_grid=param_grid)

            best_model_results = sorted(model_results, key=lambda x: x[-1])[-1]

            best_model = models[best_model_results[0]]
            
            if best_model_results[-1]<0.6:
                raise CustomException('Model performance is below threshold', sys)
            
            logging.info('Found best model')

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            return best_model_results[-1]

        except Exception as e:
            raise CustomException(e, sys)