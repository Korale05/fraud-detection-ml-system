from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from dataclasses import dataclass
import os 
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model
from src.utils import save_obj
from imblearn.over_sampling import SMOTE
@dataclass
class ModelTrainerConfi:
    train_model_path = os.path.join('artifacts','modle.pkl')

class ModleTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainerConfi()
    
    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Splitting the train and test set")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
                )
            
            logging.info(f"Train Shape {x_train.shape}")
            logging.info(f"Test shape {x_test.shape}")


            fraud_count = (y_train == 1).sum()
            non_fraud_count = (y_train == 0).sum()

            fraud_weight = non_fraud_count/fraud_count

            class_weight = {0 : 1 ,1:fraud_weight}
            
            forest = RandomForestClassifier(
                n_estimators=500,           # More trees for better performance
                max_depth=8,               # Deeper trees to capture patterns
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                class_weight=class_weight,    # Still use class weight as backup
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            logging.info("Training the randomforest ")
            forest.fit(x_train,y_train)
            logging.info("Model Training completed")

            evalution_result : dict = evaluate_model(x_train,y_train,x_test,y_test,forest)
            logging.info("Model evalution results : ")

            for metric,value in evalution_result.items():
                logging.info(f"{metric} : {value}")
            
            logging.info(f"Model is saved to {self.model_train_config.train_model_path} This path")
            save_obj(
                file_path=self.model_train_config.train_model_path,
                obj=forest
            )
            return evalution_result
        except Exception as e:
            raise CustomException(e,sys)
        