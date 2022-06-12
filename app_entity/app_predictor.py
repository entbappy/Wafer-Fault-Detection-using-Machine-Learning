import os
import sys

from app_exception import AppException
from app_util import load_object
from collections import namedtuple
import pandas as pd


class AppData:

    def __init__(self, file_path):
        try:
            self.file_path = file_path
        except Exception as e:
            raise AppException(e, sys) from e

    def get_wafer_input_data_frame(self):

        try:
            # Load the data from the file
            data_frame = pd.read_csv(self.file_path)
            return data_frame
        except Exception as e:
            raise AppException(e, sys) from e


class AppPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise AppException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise AppException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            wafer_prediction_values = model.predict(X)
            return wafer_prediction_values
        except Exception as e:
            raise AppException(e, sys) from e
    
    def save_predictions_in_csv(self, wafer_prediction_values):
        try:
            wafers = wafer_prediction_values.iloc[:,0]
            wafer_predictions = wafer_prediction_values['prediction'].tolist()
            wafer_predictions_df = pd.DataFrame(data={"wafer": wafers, "prediction": wafer_predictions})
            wafer_predictions_df.to_csv("wafer_predictions.csv", index=False)
        except Exception as e:
            raise AppException(e, sys) from e