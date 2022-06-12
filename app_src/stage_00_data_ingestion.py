from app_entity.artifact_entity import DataIngestionArtifact
from app_logger import logging
from app_exception import AppException
from app_entity import DataIngestionConfig
import sys
import os
import tarfile
from six.moves import urllib


class DataIngestion:
    

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        DataIngestion Intialization
        data_ingestion_config: DataIngestionConfig 
        """
        try:
            logging.info(f"{'='*20}Data Ingestion log started.{'='*20} ")
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise AppException(e, sys) from e

    
    def extract_tgz_file(self,tgz_file_path: str):
        """
        tgz_file_path: str
        Extracts the tgz file into the raw data directory
        Function returns None
        """
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            wafer_tgz = tarfile.open(tgz_file_path)
            os.makedirs(raw_data_dir, exist_ok=True)
            logging.info(f"Extracting tgz file: {tgz_file_path} into dir: {raw_data_dir}")
            wafer_tgz.extractall(path=raw_data_dir)
            wafer_tgz.close()
            return raw_data_dir
        except Exception as e:
            raise AppException(e,sys) from e

    def download_wafer_data(self):
        """
        Fetch wafer data from the url
        
        """
        try:
            
            wafer_file_url = self.data_ingestion_config.dataset_download_url
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir
            os.makedirs(tgz_download_dir, exist_ok=True)
            wafer_file_name = os.path.basename(wafer_file_url)
            tgz_file_path = os.path.join(tgz_download_dir, wafer_file_name)
            logging.info(f"Downloading wafer data from {wafer_file_url} into file {tgz_file_path}")
            urllib.request.urlretrieve(wafer_file_url,tgz_file_path)
            logging.info(f"Downloaded wafer data from {wafer_file_url} into file {tgz_file_path}")
            return tgz_file_path
        except Exception as e:
            raise AppException(e, sys) from e

    
    def create_good_bad_folder(self)->DataIngestionArtifact:
        try:
            data_ingestion_config = self.data_ingestion_config

            #saving the train and test dataframes   
            good_file_path = os.path.join(data_ingestion_config.ingested_good_data_dir)
            bad_file_path = os.path.join(data_ingestion_config.ingested_bad_data_dir)

            os.makedirs(data_ingestion_config.ingested_good_data_dir, exist_ok=True)
            logging.info(f"Creating good data folder: [{good_file_path}]")
            
            os.makedirs(data_ingestion_config.ingested_bad_data_dir, exist_ok=True)
            logging.info(f"Creating bad data folder: [{bad_file_path}]")
          

            data_ingestion_artifact = DataIngestionArtifact(good_file_path=good_file_path,
            bad_file_path=bad_file_path,
            is_ingested=True,
            message="Data Ingestion completed and good-bad data folders created")
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")

            return data_ingestion_artifact

        except Exception as e:
            raise AppException(e, sys) from e

    
    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        try:
            downloaded_file_path = self.download_wafer_data()
            self.extract_tgz_file(tgz_file_path=downloaded_file_path)
            logging.info(f"{'='*20}Data Ingestion log completed.{'='*20} \n\n")
            return self.create_good_bad_folder()
        except Exception as e:
            raise AppException(e, sys) from e

   


