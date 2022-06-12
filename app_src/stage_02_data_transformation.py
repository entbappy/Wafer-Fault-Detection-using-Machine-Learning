import os
import sys
import json
import pandas as pd
import numpy as np
from app_logger import logging
from app_exception import AppException
from app_entity import DataTransformationConfig, DataValidationArtifact,DataIngestionConfig
from app_config import DATASET_SCHEMA_COLUMNS_KEY, DATASET_SCHEMA_TARGET_COLUMN_KEY
from app_entity import DataTransformationArtifact
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from app_util import save_object
from imblearn.over_sampling import SMOTE
from collections import Counter



class FeatureCorrector:

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            X['Wafer'] = X['Wafer'].str[6:].astype('int64')
            return np.array(X)
            
        except Exception as e:
            raise AppException(e, sys) from e

class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_config: DataIngestionConfig,
                 data_validation_artifact: DataValidationArtifact
                 ):
        """
        data_transformation_config: DataTransformationConfig
        data_ingestion_config: DataIngestionConfig
        data_validation_artifact: DataValidationArtifact
        
        """
        try:
            logging.info(f"{'=' * 20}Data Transformation log started.{'=' * 20} ")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_config = data_ingestion_config
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise AppException(e, sys) from e

    
    def get_data_transformer(self) -> ColumnTransformer:
        try:

            # reading schema file path
            schema_file_path = self.data_validation_artifact.schema_file_path

            with open(schema_file_path, 'r') as f:
                dataset_schema = json.load(f)

            # spliting input columns and target column
            # TARGET_COLUMN
            target_column = dataset_schema[DATASET_SCHEMA_TARGET_COLUMN_KEY]

            columns = list(dataset_schema[DATASET_SCHEMA_COLUMNS_KEY].keys())
            columns.remove(target_column)

            categorical_column = []
            for column_name, data_type in dataset_schema[DATASET_SCHEMA_COLUMNS_KEY].items():
                if data_type == "category" and column_name != target_column:
                    categorical_column.append(column_name)

            numerical_column = list(
                filter(lambda x: x not in categorical_column, columns))

            num_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy="mean")),
                    ('scaler', StandardScaler())])

            cat_pipeline = Pipeline(steps=[
                ('feature_generator', FeatureCorrector())])

            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_column),
                ('cat_pipeline', cat_pipeline, categorical_column)])

            return preprocessing

        except Exception as e:
            raise AppException(e, sys) from e


    def apply_upsample(self,X,y):
        try:
            sm = SMOTE(random_state=42)
            X,y = sm.fit_resample(X,y)
            logging.info(f"Applied upsampling.")
            return X,y
        except Exception as e:
            raise AppException(e, sys) from e


    @staticmethod
    def save_numpy_array_data(file_path: str, array: np.array):
        """
        Save numpy array data to file
        file_path: str location of file to save
        array: np.array data to save
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, 'wb') as file_obj:
                np.save(file_obj, array)
        except Exception as e:
            raise AppException(e, sys) from e

    @staticmethod
    def load_numpy_array_data(file_path: str) -> np.array:
        """
        load numpy array data from file
        file_path: str location of file to load
        return: np.array data loaded
        """
        try:
            with open(file_path, 'rb') as file_obj:
                
                return np.load(file_obj)
        except Exception as e:
            raise AppException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,'wafer.csv')
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,'wafer.csv')

            schema_file_path = self.data_validation_artifact.schema_file_path

            logging.info(f"train file path: [{train_file_path}]\n \
            test file path: [{test_file_path}]\n \
            schema_file_path: [{schema_file_path}]\n. ")

            # loading the dataset
            logging.info(f"Loading train and test dataset...")
            train_dataframe = pd.read_csv(train_file_path)

            test_dataframe = pd.read_csv(test_file_path)
            logging.info("Data loaded successfully.")

            with open(schema_file_path, 'r') as f:
                dataset_schema = json.load(f)
            
            target_column_name = dataset_schema[DATASET_SCHEMA_TARGET_COLUMN_KEY]
            logging.info(f"Target column name: [{target_column_name}].")

            # target_column
            logging.info(f"Converting target column into numpy array.")
            train_target_arr = np.array(train_dataframe[target_column_name])
            logging.info('Before applying upsample for training {}'.format(Counter(train_target_arr)))
            test_target_arr = np.array(test_dataframe[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")

            # dropping target column from the dataframe
            logging.info(f"Dropping target column from the dataframe.")
            train_dataframe.drop(target_column_name, axis=1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")

            logging.info(f"Creating preprocessing object.")
            preprocessing = self.get_data_transformer()
            logging.info(f"Creating preprocessing object completed.")
            logging.info(f"Preprocessing object learning started on training dataset.")
            logging.info(f"Transformation started on training dataset.")
            train_input_arr = preprocessing.fit_transform(train_dataframe)
            logging.info(f"Preprocessing object learning completed on training dataset.")
            logging.info('Before applying upsample X_train shape {}'.format(train_input_arr.shape))

            logging.info(f"Transformation started on testing dataset.")
            test_input_arr = preprocessing.transform(test_dataframe)
            logging.info(f"Transformation completed on testing dataset.")

            #applying upsample for handling imbalenced data
            train_input_arr, train_target_arr = self.apply_upsample(train_input_arr, train_target_arr)
            logging.info('After applying upsample for training {}'.format(Counter(train_target_arr)))
            logging.info('After applying upsample X_train shape {}'.format(train_input_arr.shape))

            # adding target column back to the numpy array
            logging.info("Started concatenation of target column back  into transformed numpy array.")
            train_arr = np.c_[train_input_arr, train_target_arr]
            logging.info('Train array shape {}'.format(train_arr.shape))
            test_arr = np.c_[test_input_arr, test_target_arr]
            logging.info("Completed concatenation of  target column back  into transformed numpy array.")

            # generating file name such as wafer_transformed.npy
            file_name = os.path.basename(train_file_path)
            file_extension_starting_index = file_name.find(".")
            file_name = file_name[:file_extension_starting_index]
            file_name = file_name + "_transformed.npy"
            logging.info(f"File name: [{file_name}] for transformed dataset.")

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            transformed_train_file_path = os.path.join(transformed_train_dir, file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, file_name)
            logging.info(f"Transformed train file path: [{transformed_train_file_path}].")
            logging.info(f"Transformed test file path: [{transformed_test_file_path}].")
            # saving the transformed data 
            logging.info(f"Saving transformed train and test dataset to file.")
            DataTransformation.save_numpy_array_data(file_path=transformed_train_file_path,
                                                     array=train_arr)

            DataTransformation.save_numpy_array_data(file_path=transformed_test_file_path,
                                                     array=test_arr)
            logging.info(f"Saving transformed train and test dataset to file completed.")

            logging.info(f"Saving preprocessing object")
            preprocessed_object_file_path = self.data_transformation_config.preprocessed_object_file_path
            # saving the preprocessed object
            save_object(file_path=preprocessed_object_file_path,
                        obj=preprocessing)
            logging.info(f"Saving preprocessing object in file: [{preprocessed_object_file_path}] completed.")
            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
                                                                      message="Data transformed successfully",
                                                                      transformed_train_file_path=transformed_train_file_path,
                                                                      transformed_test_file_path=transformed_test_file_path,
                                                                      preprocessed_object_file_path=preprocessed_object_file_path)
            logging.info(f"Data Transformation artifact: [{data_transformation_artifact}] created successfully")
            logging.info(f"{'=' * 20}Data Transformation log ended.{'=' * 20} ")
            return data_transformation_artifact

        except Exception as e:
            raise AppException(e, sys) from e

