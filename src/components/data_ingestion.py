# It contain all the data that we bring from some file or a database .

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) # index False means dont include the row names in the new dataframe and header True means include columns name in the new data frame.

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42) # test_size=0.2 means test data set will have 20% of the total data and train data set will have 80% of the total data.
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                # Answer to the question of why are we returning the paths of train and test dataset that are created :This design pattern ensures that the data ingestion process is cleanly separated from the rest of the pipeline, and the paths to the data files are provided to whatever needs them next in the process.

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion() # Object of the class DataIngestion is created .
    obj.initiate_data_ingestion() # Newly created object of DataIngestion class is calling the initiate_data_ingestion method of DataIngestion class .
    
    
    train_data,test_data=obj.initiate_data_ingestion() # train_data and test_data contains the path where train data and test data is stored respectively.

    data_transformation=DataTransformation() # Created the object of the class DataTransformation named data_transformation .
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))  # prints the r2_square of the best ML model .


