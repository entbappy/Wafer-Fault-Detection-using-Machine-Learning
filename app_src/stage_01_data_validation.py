from app_logger import logging
from app_exception import AppException
from app_entity import DataValidationConfig, DataIngestionArtifact, DataValidationArtifact, DataIngestionConfig
import os
import sys
import re
import json
import shutil
import pandas as pd
import glob
from sklearn.model_selection import train_test_split


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'='*20}Data Validation log started.{'='*20} ")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AppException(e, sys) from e

    
    def values_from_schema(self):
        try:
            with open(self.data_validation_config.schema_file_path, 'r') as f:
                dic = json.load(f)
                f.close()
            pattern = dic['SampleFileName']
            lengthOfDateStampInFile = dic['LengthOfDateStampInFile']
            lengthOfTimeStampInFile = dic['LengthOfTimeStampInFile']
            column_names = dic['ColName']
            number_of_columns = dic['NumberofColumns']

            return lengthOfDateStampInFile, lengthOfTimeStampInFile, column_names, number_of_columns

        except Exception as e:
            raise AppException(e, sys) from e


    def manual_regex_creation(self):
        """
                                Method Name: manual_regex_creation
                                Description: This method contains a manually defined regex based on the "FileName" given in "Schema" file.
                                            This Regex is used to validate the filename of the training data.
                                Output: Regex pattern
                                On Failure: None

                                Written By: iNeuron Intelligence
                                        """
        regex = "['wafer']+['\_'']+[\d_]+[\d]+\.csv"
        return regex

    
    def validation_file_name_raw(self,regex,LengthOfDateStampInFile,LengthOfTimeStampInFile):
       
        onlyfiles = [f for f in os.listdir(self.data_ingestion_config.raw_data_dir)]

        try:
            for filename in onlyfiles:
                if (re.match(regex, filename)):
                    splitAtDot = re.split('.csv', filename)
                    splitAtDot = (re.split('_', splitAtDot[0]))
                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:
                            shutil.copy(os.path.join(self.data_ingestion_config.raw_data_dir ,filename), self.data_ingestion_artifact.good_file_path)
                            logging.info(f"Valid File name!! File moved to Good Data Folder :{filename}")

                        else:
                            shutil.copy(os.path.join(self.data_ingestion_config.raw_data_dir ,filename), self.data_ingestion_artifact.bad_file_path)
                            logging.info(f"Invalid File Name!! File moved to Bad Raw Folder :{filename}")
                    else:
                        shutil.copy(os.path.join(self.data_ingestion_config.raw_data_dir ,filename), self.data_ingestion_artifact.bad_file_path)
                        logging.info(f"Invalid File Name!! File moved to Bad Raw Folder :{filename}")
                else:
                    shutil.copy(os.path.join(self.data_ingestion_config.raw_data_dir ,filename), self.data_ingestion_artifact.bad_file_path)
                    logging.info(f"Invalid File Name!! File moved to Bad Raw Folder :{filename}")

        except Exception as e:
            raise AppException(e, sys) from e

    
    def validate_column_length(self,NumberofColumns):
        try:
            for file in os.listdir(self.data_ingestion_artifact.good_file_path):
                csv = pd.read_csv(os.path.join(self.data_ingestion_artifact.good_file_path, file))
                if csv.shape[1] == NumberofColumns:
                    pass
                else:
                    shutil.move(os.path.join(self.data_ingestion_artifact.good_file_path, file), self.data_ingestion_artifact.bad_file_path)
                    logging.info(f"Invalid Column Length for the file!! File moved to Bad Raw Folder :{file}")
        
        except Exception as e:
            raise AppException(e, sys) from e

    
    def validate_missing_values_in_whole_column(self):
        try:
            for file in os.listdir(self.data_ingestion_artifact.good_file_path):
                csv = pd.read_csv(os.path.join(self.data_ingestion_artifact.good_file_path, file))
                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count+=1
                        shutil.move(os.path.join(self.data_ingestion_artifact.good_file_path, file),self.data_ingestion_artifact.bad_file_path)
                        logging.info(f"Invalid !! Missing value in whole column, File moved to Bad Raw Folder :{file}")
                        break
                if count==0:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    csv.to_csv(os.path.join(self.data_ingestion_artifact.good_file_path, file), index=None, header=True)

        except Exception as e:
            raise AppException(e, sys) from e

    
    def data_validation(self) -> bool:
        try:
            lengthOfDateStampInFile, lengthOfTimeStampInFile, column_names, number_of_columns = self.values_from_schema()
            # getting the regex defined to validate filename
            regex = self.manual_regex_creation()
            
            self.validation_file_name_raw(regex, lengthOfDateStampInFile, lengthOfTimeStampInFile)
            # validating column length in the file
            self.validate_column_length(number_of_columns)
            # validating if any column has all values missing
            self.validate_missing_values_in_whole_column()

            return True
        
        except Exception as e:
            raise AppException(e, sys) from e



    def merged_all_csv_from_good_folder(self):
        try:
            os.makedirs(self.data_ingestion_config.merged_csv_dir, exist_ok=True)

            good_data_path = self.data_ingestion_artifact.good_file_path
            all_files = glob.glob(os.path.join(good_data_path, "*.csv"))    
            df_from_each_file = (pd.read_csv(f) for f in all_files)
            concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
            concatenated_df.to_csv(os.path.join(self.data_ingestion_config.merged_csv_dir, "merged_data.csv"), index=None, header=True)
            logging.info(f"Data merged to merged_data.csv")

        except Exception as e:
            raise AppException(e, sys) from e
    

    def train_test_split_data(self):
        try:
            os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
            os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
            df = pd.read_csv(os.path.join(self.data_ingestion_config.merged_csv_dir, "merged_data.csv"))
            X_train, X_test = train_test_split(df, test_size=0.33, random_state=42)
            X_train.to_csv(os.path.join(self.data_ingestion_config.ingested_train_dir, "wafer.csv"), index=None, header=True)
            X_test.to_csv(os.path.join(self.data_ingestion_config.ingested_test_dir, "wafer.csv"), index=None, header=True)
            logging.info(f"Data splitted into train & test")

        except Exception as e:
            raise AppException(e, sys) from e



    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            status = self.data_validation()
            message = "Data validation status: {}".format(status)
            self.merged_all_csv_from_good_folder()
            self.train_test_split_data()
            data_validation_artifact = DataValidationArtifact(is_validated=status,
                                                              message=message,
                                                              schema_file_path=self.data_validation_config.schema_file_path,
                                                              )
            logging.info(f"Data validation status: {status}.")
            logging.info(
                f"Data validation artifact: {data_validation_artifact}.")
            logging.info(f"{'='*20}Data Validation log ended.{'='*20} ")
            return data_validation_artifact

        except Exception as e:
            raise AppException(e, sys) from e

