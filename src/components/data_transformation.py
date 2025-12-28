import os
import sys

#adding project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,project_root)

from src.exception import CustomException
from src.logger import logging

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    """
    This function is responsible for all data transformation happen in the data 
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def create_user_feature(self,df,is_Training=True):
        """Creating user based feature before removing User_Id"""
        """We creating this feature but when we take input from the user we are taking as 1 data
        point so we can not calculate the  
                user_transaction_count → Need all user's past transactions
                user_fraud_rate → Need user's fraud history
                user_avg_amount → Need user's past transactions
                days_since_first_transaction → Need user's first transaction date
        """
        """To solve this problem what we do we store the User stastics during Training and load
            This statistics during traning """
        
        try : 
            # Convert Timestamp to datetime first
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            if is_Training:
                self.user_state={
                    'user_transaction_count' : df.groupby('User_ID').size().to_dict(),
                    'user_avg_amount' : df.groupby('User_ID')['Transaction_Amount'].mean().to_dict(),
                    'user_amount_std' : df.groupby('User_ID')['Transaction_Amount'].std().fillna(0).to_dict(),
                    'user_first_transaction' : df.groupby('User_ID')['Timestamp'].min().to_dict()
                }
            # Apply feature
            df['user_transaction_count'] = df['User_ID'].map(
                self.user_state.get('user_transaction_count',{})
            ).fillna(1)

            df['user_avg_amount'] = df['User_ID'].map(
                self.user_state.get('user_avg_amount',{})
            ).fillna(df['Transaction_Amount'])
            
            df['user_amount_std'] = df['User_ID'].map(
                self.user_state.get('user_amount_std',{})
            ).fillna(0)

            # Deviation from user's normal behavior
            df['amount_deviation_from_user_avg'] = (df['Transaction_Amount'] - df['user_avg_amount']) / (df['user_amount_std']+1e-5)
            df['amount_deviation_from_user_avg']  = df['amount_deviation_from_user_avg'].fillna(0)

            df['user_first_tnx_data'] = df['User_ID'].map(
                self.user_state.get('user_first_transaction',{})
            ).fillna(df['Timestamp'])

            df['days_since_first_transaction'] = ((df['Timestamp'] - df['user_first_tnx_data']).dt.total_seconds()/86400).fillna(0)

            df = df.drop('user_first_tnx_data',axis=1)

            return df
        
        except Exception as e:

            raise CustomException(e,sys)
        
    def create_time_feature(self,df):
        """Extract TIme feature"""
        try :
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Hour'] = df['Timestamp'].dt.hour
            df['day_of_month'] = df['Timestamp'].dt.day
            df['week'] = df['Timestamp'].dt.dayofweek
            return df
        except Exception as e:
            raise CustomException(e,sys)
    
    def remove_unnecessary_column(self,df):
        """Remove unnecessary column like 
                                    Unser_ID we created 4 feature of it
                                    Transaction_ID
                                    Times
                                    tamp
                                    """
        try : 
            column_to_drop = ['Transaction_ID','User_ID','Timestamp']

            df = df.drop(columns = column_to_drop)

            logging.info("Un necessary coloumn is removed")

            return df
        
        except Exception as e:

            raise CustomException(e,sys)
    
    def get_data_transform_obj(self,df):
        try : 
            
            cat_feature = df.select_dtypes(include='object').columns.tolist()
            num_feature = df.select_dtypes(exclude='object').columns.tolist()

            num_feature = [col for col in num_feature if col !='Fraud_Label']
            
            logging.info(f"Categorical_feature : {cat_feature}")

            logging.info(f"Numerical Feature : {num_feature}")


            num_pipline = Pipeline(
                steps=[
                    ('Simple imputer',SimpleImputer(strategy='median')),
                    ("Standard Scaler",StandardScaler())
                ]
            )
            cat_pipline = Pipeline(
                steps=[
                    ("simple Imputer",SimpleImputer(strategy='most_frequent')),
                    ("One hot encoder",OneHotEncoder(drop='first',handle_unknown='ignore'))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipline',num_pipline,num_feature),
                    ("cat pipline",cat_pipline,cat_feature)
                ]
            )

            logging.info("Preprocessing pipline is created")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            #Read the data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading Traning and Test data")

            #Creating featur for train data
            train_df = self.create_user_feature(train_df,is_Training=True)
            train_df = self.create_time_feature(train_df)
            train_df = self.remove_unnecessary_column(train_df)

            #Create feature for test data
            test_df = self.create_user_feature(test_df,is_Training=True)
            test_df = self.create_time_feature(test_df)
            test_df = self.remove_unnecessary_column(test_df)

            logging.info("Feature engineerig is completed")

            #GEt preprocessing obj after feature engineering
            preprocessing_obj = self.get_data_transform_obj(train_df)
            target_column_name = 'Fraud_Label'

            #Seperate feature and target
            input_feature_train_df = train_df.drop(target_column_name,axis=1)
            target_feature_train_df = train_df[target_column_name] 

            input_feature_test_df= test_df.drop(target_column_name,axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object to train and test data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #Combine the features and target
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("Data Transformation is Done successfully")
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)



    
        
