import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run(self):
        try:
            logging.info("Starting training pipeline")
            
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            
            train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            
            r2_square = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            logging.info(f"Training pipeline completed successfully. R2 Score: {r2_square}")

        except Exception as e:
            logging.error(f"Training pipeline failed: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run()
