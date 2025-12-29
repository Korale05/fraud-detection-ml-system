import os
import sys

#adding project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,project_root)


from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.model_trainer import ModleTrainer
from src.components.data_transformation import DataTransformation,DataTransformationConfig
@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join("artifacts","train.csv")
    test_data_path : str = os.path.join("artifacts","test.csv")
    raw_data_path : str = os.path.join("artifacts",'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.inguestion_config = DataIngestionConfig()
    def remove_lekage_column(slef,df):
        """
        Remove columns that contain data leakage
        """
        try:
            leakage_columns = [
                'Previous_Fraudulent_Activity',  # Future information
                'Failed_Transaction_Count_7d',   # Includes future transactions
                'Risk_Score',                     # Calculated using fraud labels
                'IP_Address_Flag'                # Likely calculated using fraud patterns
            ]
            df = df.drop(columns = leakage_columns)
            return df
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_ingestion(self):
        logging.info("Inititate the data ingestion")
        try:
            df = pd.read_csv(r"notebook\data\dataset.csv")

            logging.info("Removing Lekage column")

            df = self.remove_lekage_column(df)

            logging.info("Converting Raw data into data frame")

            os.makedirs(os.path.dirname(self.inguestion_config.train_data_path),exist_ok=True)

            logging.info("Artifacts Directory is created")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            logging.info("Train test split is applied")

            train_set.to_csv(self.inguestion_config.train_data_path,index = False,header = True)

            test_set.to_csv(self.inguestion_config.test_data_path,index = False,header = True)

            logging.info("Train_set and Test_set is saved in artifacts folder")

            df.to_csv(self.inguestion_config.raw_data_path,index=False,header=True)

            logging.info("Raw data also saved in artifacts folder")

            return (
                self.inguestion_config.train_data_path,
                self.inguestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()
    obj2= DataTransformation()
    train_arr,test_arr,_ = obj2.initiate_data_transformation(train_path,test_path)
    obj3 = ModleTrainer()
    evaluation_report = obj3.initiate_model_training(train_arr,test_arr)