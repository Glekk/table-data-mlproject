import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_ingestion import DataIngestion 
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def train(self):
        try:
            data_ingestion = DataIngestion()
            train_data, test_data = data_ingestion.initiate_data_ingestion()

            data_transformation = DataTransformation()
            train_arr, test_arr,_ = data_transformation.initiate_data_transormation(train_data, test_data)

            model_trainer = ModelTrainer()  
            print(model_trainer.initiate_model_trainer(train_arr, test_arr))
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    tp = TrainPipeline()
    tp.train()