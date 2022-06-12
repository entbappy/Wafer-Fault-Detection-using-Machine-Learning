from app_exception import AppException
from collections import namedtuple

DataIngestionConfig = namedtuple("DatasetConfig", ["dataset_download_url",
                                                   "raw_data_dir",
                                                   "tgz_download_dir",
                                                   "ingested_good_data_dir",
                                                   "ingested_bad_data_dir",
                                                   "merged_csv_dir",
                                                   "ingested_train_dir",
                                                   "ingested_test_dir"])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])

DataValidationConfig = namedtuple("DataValidationConfig", ["schema_file_path"])

DataTransformationConfig = namedtuple("DataTransformationConfig", ["transformed_train_dir", 
                                                                   "transformed_test_dir",
                                                                   "preprocessed_object_file_path"])


ModelTrainerConfig = namedtuple("ModelTrainerConfig", ["trained_model_file_path","clustering_model_dir","base_accuracy"])

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig", ["model_evaluation_file_path","time_stamp"])

ModelPusherConfig = namedtuple("ModelPusherConfig", ["export_dir_path"])