from flask import Flask, request
from flask import send_file, abort, render_template
import os

from app_entity.app_predictor import AppData
from app_entity.app_predictor import AppPredictor


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "wafer"
SAVED_MODELS_DIR_NAME = "saved_models"
PREDICTION_FOLDER_NAME = "prediction_folder"
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)
PREDICTION_FILE_DIR = os.path.join(ROOT_DIR, PREDICTION_FOLDER_NAME)



HOUSING_DATA_KEY = "housing_data"
MEDIAN_HOUSING_VALUE_KEY = "median_house_value"

app = Flask(__name__)




@app.route('/artifact', defaults={'req_path': 'wafer'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("wafer", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file):file for file in os.listdir(abs_path)}
    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
         "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/train', methods=['GET', 'POST'])
def train():
    from subprocess import call
    return_code = call(["python", "test.py"])
    print(return_code)
    return render_template('train.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    from subprocess import call
    # return_code = call(["python", "test.py"])
    # print(return_code)

    # Upload the file to the server
    if request.method == 'POST':
        f = request.files['fileupload']
        os.makedirs(PREDICTION_FILE_DIR, exist_ok=True)
        file_path = os.path.join(PREDICTION_FILE_DIR, f.filename)
        f.save(file_path)

        app_data = AppData(file_path)
        wafer_df= app_data.get_wafer_input_data_frame()
        # print(wafer_df.head())
        wafer_predictor = AppPredictor(model_dir=MODEL_DIR)
        wafer_prediction_values = wafer_predictor.predict(X=wafer_df)
        wafer_predictor.save_predictions_in_csv(wafer_prediction_values)
        
    return render_template('predict.html')

@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file):file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route('/logs', defaults={'req_path': 'logs'})
@app.route('/logs/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs("logs", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file):file for file in os.listdir(abs_path)}
    
    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)



if __name__ == '__main__':
    app.run()
