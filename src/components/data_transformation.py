# Here we do like filling the missing values with median,etc and then converting categorical features into numerical features, one hot encoding, etc. And doing standardisation and normalisation. 

import sys  # It handles the error. Helps in Exception Handling.
from dataclasses import dataclass  # It is used to call __init__ method of the class.

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer # It helps in creating the pipelines .
from sklearn.impute import SimpleImputer # It is used to handle missing values in the dataframe .
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object   #  Saves the pickle file in the hard disk.

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl") # If I create a model and want to save it in a pickle file, for that I am creating it.

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self): # This fn converts categorical features into numerical features.
        '''
        This function is responsible for data transformation
        
        '''
        try:
            # math_score which is the target here is excluded here.
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(  # For missing values in the data, we are creating a pipeline. The startegy(plan) is to replace missing numerical values with median. This pipeline run on training dataset.
                steps=[
                ("imputer",SimpleImputer(strategy="median")),  # imputer is responsible for handling my missing values.
                ("scaler",StandardScaler())   # Standardisation of numerical features occur here. 

                ]
            )
 
            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")), # imputer handles missing values. The strategy(plan) is to replace missing values with most_frequent value ie mode. 
                ("one_hot_encoder",OneHotEncoder()), # One hot encoder converts cat. features into numerical features.
                ("scaler",StandardScaler(with_mean=False)) # Now after converting cat. features into num. features, we perform standardisation on them. 
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")  # This message goes into the log file.
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer( # We use this Column Transform to combine numerical pipeline with categorical pipeline. So that thins are executed in a sequence.
                [
                ("num_pipeline",num_pipeline,numerical_columns), # In this line, we 1st give the name of pipeline then we give actual numerical pipeline and then we give the numerical columns of df. 
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object() # This preprocessing object needs to be converted into a pickle file . 

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            # np.c_ : concatenates the two arrays column-wise.
            # target_feature_train_df and target_feature_test_df are the target arrays for training and testing data respectively.
            # np.array(target_feature_train_df) and np.array(target_feature_test_df) convert the target dataframes to numpy arrays.
            # train_arr is the combined array of training features and target.
            # test_arr is the combined array of testing features and target.
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(  # It is called from the utils.py to save the pickle file into the harddisk. The pickle file will contain the object in the form of binary values ie a data stream so that the object can be stored in a hard disk or can be transmitted.

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
