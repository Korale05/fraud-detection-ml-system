import os
import sys


#adding project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,project_root)
from src.utils import load_pickle
import pickle
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
    user_state_file_path = os.path.join('artifacts','user_state.pkl')

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
            df = df.sort_values("Timestamp")
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            user_stats = {}

            df['user_transaction_count'] = 0.0
            df['user_avg_amount'] = 0.0
            df['user_amount_std'] = 0.0
            df['days_since_first_transaction'] = 0.0
            df['is_unusual_spend'] = 0.0

            for i, row in df.iterrows():
                user = row['User_ID']
                amount = row['Transaction_Amount']
                t = row['Timestamp']
                
                if user not in user_stats:
                    # first ever transaction
                    user_stats[user] = {
                        "amounts": [amount],
                        "first_ts": t
                    }
                    
                    df.at[i, 'user_transaction_count'] = 1
                    df.at[i, 'user_avg_amount'] = amount
                    df.at[i, 'user_amount_std'] = 0
                    df.at[i, 'days_since_first_transaction'] = 0
                
                else:
                    history = user_stats[user]["amounts"]
                    first_ts = user_stats[user]["first_ts"]
                    
                    df.at[i, 'user_transaction_count'] = len(history) + 1
                    df.at[i, 'user_avg_amount'] = np.mean(history)
                    df.at[i, 'user_amount_std'] = np.std(history)
                    df.at[i, 'days_since_first_transaction'] = (t - first_ts).total_seconds() / 86400
                    
                    # update history AFTER using it
                    history.append(amount)
            df['amount_deviation_from_user_avg'] = (
                (df['Transaction_Amount'] - df['user_avg_amount']) / 
                (df['user_amount_std'] + 1e-5)
            )
            df['is_unusual_spend'] = (df['amount_deviation_from_user_avg'] > 3).astype(int)

            final_user_state = {
                user: {
                    "avg": np.mean(stats["amounts"]),
                    "std": np.std(stats["amounts"]),
                    "count": len(stats["amounts"]),
                    "first_ts": stats["first_ts"]
                }
                for user, stats in user_stats.items()
            }

            with open("artifacts/user_state.pkl", "wb") as f:
                pickle.dump(final_user_state, f)

            return(
                df
            )
        except Exception as e:

            raise CustomException(e,sys)
    def create_user_features_inference(self, df):
        df = df.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        user_state = load_pickle(self.data_transformation_config.user_state_file_path)

        df['user_transaction_count'] = 0.0
        df['user_avg_amount'] = 0.0
        df['user_amount_std'] = 0.0
        df['days_since_first_transaction'] = 0.0
        df['amount_deviation_from_user_avg'] = 0.0
        for i ,row in df.iterrows():
            user = row['User_ID']
            amount = row['Transaction_Amount']
            t = row['Timestamp']

            if user in user_state:
                s = user_state[user]
                count = s["count"]
                avg = s["avg"]
                std = s["std"]
                first_ts = s["first_ts"]
                days = (t - first_ts).total_seconds() / 86400
            else:
                count = 1
                avg = amount
                std = 0
                days = 0

            deviation = (amount - avg) / (std + 1e-5)
            df.loc[i,'user_transaction_count'] = count
            df.loc[i, 'user_transaction_count'] = count
            df.loc[i, 'user_avg_amount'] = avg
            df.loc[i, 'user_amount_std'] = std
            df.loc[i, 'days_since_first_transaction'] = days
            df.loc[i, 'amount_deviation_from_user_avg'] = deviation
        df['is_unusual_spend'] = (df['amount_deviation_from_user_avg'] > 3).astype(int)
        return df

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
            cat_feature = list()
            num_feature = list()

            for i in df.columns:
                if i == 'Fraud_Label':
                    continue
                if df[i].nunique() <= 15:
                    cat_feature.append(i)
                else:
                    num_feature.append(i)

            cat_feature =[col for col in cat_feature if col !='Fraud_Label']
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
                ],
                sparse_threshold=0
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
            train_df = self.create_user_feature(train_df)
            train_df = self.create_time_feature(train_df)
            train_df = self.remove_unnecessary_column(train_df)

            #Create feature for test data
            test_df = self.create_user_features_inference(test_df)
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



    
        
