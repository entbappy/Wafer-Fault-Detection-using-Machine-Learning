from app_entity.artifact_entity import DataTransformationArtifact
from app_logger import logging
from app_exception import AppException
from app_entity import ModelEvaluationConfig, ModelTrainerArtifact, DataValidationArtifact, \
    ModelEvaluationArtifact, DataIngestionConfig,MetricInfoArtifact
from app_config import DATASET_SCHEMA_TARGET_COLUMN_KEY
import numpy as np
import os
import json
import pandas as pd
import sys
from app_util import write_yaml_file, read_yaml_file, load_object
from app_src import DataTransformation, ModelTrainer
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics  import roc_auc_score,accuracy_score

BEST_MODEL_KEY = "best_model"
HISTORY_KEY = "history"
MODEL_PATH_KEY = "model_path"


class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_config: DataIngestionConfig,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info(f"{'=' * 20}Model Evaluation log started.{'=' * 20} ")
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_config = data_ingestion_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise AppException(e, sys) from e

    def get_best_model(self):
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path

            print(f"model_evaluation_file_path: {model_evaluation_file_path}")
            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path,
                                )
                return model
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)

            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content
            print(f"model_eval_file_content: {model_eval_file_content}")

            if BEST_MODEL_KEY not in model_eval_file_content:
                return model

            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
        except Exception as e:
            raise AppException(e, sys) from e

    
    def evaluate_model(self, model_list: list, X_train, y_train, X_test, y_test, base_accuracy) -> MetricInfoArtifact:
        try:
            index_number = 0
            metric_info_artifact = None
            for model in model_list:
                model_name = str(model)
                logging.info(
                    f"Started evaluating model: [{type(model).__name__}]")
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                y_train_pred = y_train_pred['prediction'].values.astype(int)
                y_test_pred = y_test_pred['prediction'].values.astype(int)
                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_test_pred)
            
                # Calculating harmonic mean of train_accuracy and test_accuracy
                model_accuracy = (2 * (train_acc * test_acc)) / (train_acc + test_acc)
                diff_test_train_acc = abs(test_acc - train_acc)
                
                train_precision, train_recall, train_fscore, train_support = score(y_train, y_train_pred)
                test_precision, test_recall, test_fscore, test_support = score(y_test, y_test_pred)

                message = f"{'*' * 20}{model_name} metric info{'*' * 20}"
                logging.info(f"{message}")
                message = f"\n\t\tTrain accuracy: [{train_acc}]."
                message += f"\n\t\tTest accuracy: [{test_acc}]."
                message += f"\n\t\tTrain precision: [{train_precision}]."
                message += f"\n\t\tTrain recall: [{train_recall}]."
                message += f"\n\t\tTrain fscore: [{train_fscore}]."
                message += f"\n\t\tTest precision: [{test_precision}]."
                message += f"\n\t\tTest recall: [{test_recall}]."
                message += f"\n\t\tTest fscore: [{test_fscore}]."
                message += f"\n\t\tModel accuracy: [{model_accuracy}]."
                message += f"\n\t\tBase accuracy: [{base_accuracy}]."
                message += f"\n\t\tDiff test train accuracy: [{diff_test_train_acc}]."
               
                logging.info(message)
                message = f"{'*' * 20}{model_name} metric info{'*' * 20}"
                logging.info(message)

                if model_accuracy >= base_accuracy and diff_test_train_acc < 0.06:
                    base_accuracy = model_accuracy
                    metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                              model_object=model,
                                                              train_precision=train_precision,
                                                              train_recall=train_recall,
                                                              train_fscore=train_fscore,
                                                              test_precision=test_precision,
                                                              test_recall=test_recall,
                                                              test_fscore=test_fscore,
                                                              train_accuracy=train_acc,
                                                              test_accuracy=test_acc,
                                                              model_accuracy=model_accuracy,
                                                              index_number=index_number)

                    logging.info(
                        f"Acceptable model found {metric_info_artifact}. ")
                index_number += 1

            if metric_info_artifact is None:
                logging.info(
                    f"No model found with higher accuracy than base accuracy")

            return metric_info_artifact

        except Exception as e:
            raise AppException(e, sys) from e

    def update_evaluation_report(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_eval_content = read_yaml_file(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content
            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]

            logging.info(f"Previous eval result: {model_eval_content}")
            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path,
                }
            }

            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)

            model_eval_content.update(eval_result)
            logging.info(f"Updated eval result:{model_eval_content}")
            write_yaml_file(file_path=eval_file_path, data=model_eval_content)

        except Exception as e:
            raise AppException(e, sys) from e
    
    def initiate_model_evaluation(self):
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(file_path=os.path.join(trained_model_file_path))
            logging.info(f"Type of trained model object: {type(trained_model_object)}")
            

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,'wafer.csv')
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,'wafer.csv')

            schema_file_path = self.data_validation_artifact.schema_file_path

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
            test_target_arr = np.array(test_dataframe[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")

            # dropping target column from the dataframe
            logging.info(f"Dropping target column from the dataframe.")
            train_dataframe.drop(target_column_name, axis=1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")

            model = self.get_best_model()

            if model is None:
                logging.info("Not found any existing model. Hence accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact
            list_of_model = [model, trained_model_object]
            metric_info_artifact = self.evaluate_model(model_list=list_of_model,
                                                        X_train=train_dataframe,
                                                        y_train=train_target_arr,
                                                        X_test=test_dataframe,
                                                        y_test=test_target_arr,
                                                        base_accuracy= 0.5,
                                                        )
            logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")

            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path
                                                   )
                logging.info(response)
                return response
            
            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
            
            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact
        except Exception as e:
            raise AppException(e, sys) from e



