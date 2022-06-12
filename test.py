from app_logger import logging
from app_pipeline import TrainingPipeline
from app_util import load_object
import pandas as pd
if __name__ == '__main__':
    try:
        TrainingPipeline().start_training_pipeline()
        #model()
    except Exception as e:
        logging.info(e)
