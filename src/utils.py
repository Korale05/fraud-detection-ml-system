import pandas as pd
import numpy as np
import os
import sys
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,roc_auc_score
    ,precision_score,recall_score,f1_score
)

def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file:
            dill.dump(obj,file)

    except Exception as e:
        raise CustomException(e,sys)

import pickle

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def evaluate_model(x_train,y_train,x_test,y_test,model):
    """Evaluate the model fouces on recall for fraud detection """
    try :
        report={}
        y_test_pred = model.predict(x_test)
        y_test_proba= model.predict_proba(x_test)


         # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        class_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        accuracy = accuracy_score(y_test, y_test_pred)
        roc_score = roc_auc_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        print("Model Evaluation Result")
        print('confussion Matrix')
        print(conf_matrix)
        print('classification report')
        print(class_report)
        print('accuracu score')
        print(accuracy)
        print('roc_aoc_curve')
        print(roc_score)
        print('precision score')
        print(precision)
        print('recall')
        print(recall)
        print('f1')
        print(f1)
        # Check if recall is low and suggest improvements
        if recall < 0.7:
            print("\n⚠️ WARNING: Recall is below 70%")
            print("Suggestions to improve recall:")
            print("1. Adjust decision threshold (lower from 0.5 to 0.3-0.4)")
            print("2. Increase class_weight for fraud class")
            print("3. Use SMOTE or other sampling techniques")
            print("4. Try threshold tuning with y_test_proba")
        
        print("="*50 + "\n")
        
        # Return metrics as dictionary
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_score,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        return results


    except Exception as e:
        raise CustomException(e,sys)