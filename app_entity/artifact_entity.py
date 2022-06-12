from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact", [
    "good_file_path", "bad_file_path", "is_ingested", "message"])

DataValidationArtifact = namedtuple("DataValidationArtifact", [
    "is_validated", "message", "schema_file_path"])

DataTransformationArtifact = namedtuple("DataTransformationArtifact", ["is_transformed",
                                                                       "message", "transformed_train_file_path",
                                                                       "transformed_test_file_path",
                                                                       "preprocessed_object_file_path"])


MetricInfoArtifact = namedtuple("MetricInfo",
                                ["model_name", "model_object", "train_precision", "train_recall", "train_fscore",
                                 "test_precision", "test_recall", "test_fscore", "train_accuracy","test_accuracy","model_accuracy",
                                 "index_number"])


ModelTrainerArtifact = namedtuple("ModelTrainerArtifact", ["is_trained", "message", "trained_model_file_path"])

ModelEvaluationArtifact = namedtuple("ModelEvaluationArtifact", ["is_model_accepted", "evaluated_model_path"])

ModelPusherArtifact = namedtuple("ModelPusherArtifact", ["is_model_pusher", "export_model_file_path"])
