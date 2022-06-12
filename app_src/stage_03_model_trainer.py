from jinja2 import ModuleLoader
from app_logger import logging
from app_exception import AppException
from app_entity import ModelTrainerConfig, ModelTrainerArtifact
from app_entity import DataTransformationArtifact, MetricInfoArtifact
from app_src import DataTransformation
import sys
import os
import numpy as np
import pandas as pd
from app_util import load_object, save_object
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split

class Model_finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """

    def __init__(self):
        self.rf = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')

    def get_best_params_for_random_forest(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        logging.info('Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            # self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
            #                    "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            self.param_grid = {"n_estimators": [10], "criterion": ['gini',],
                               "max_depth": range(2, 3, 1), "max_features": ['auto',]}


            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.rf, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters

            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.rf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # training the mew model
            self.rf.fit(train_x, train_y)
            logging.info('Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.rf

        except Exception as e:
            raise AppException(e, sys) from e

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Written By: iNeuron Intelligence
                                        Version: 1.0
                                        Revisions: None

                                """
        logging.info('Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            # self.param_grid_xgboost = {

            #     'learning_rate': [0.5, 0.1, 0.01, 0.001],
            #     'max_depth': [3, 5, 10, 20],
            #     'n_estimators': [10, 50, 100, 200]

            # }
            self.param_grid_xgboost = {

                'learning_rate': [0.5],
                'max_depth': [3],
                'n_estimators': [10]

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            """Testing by just passing values 

                Returns:
                    _type_: ndarray of shape (n_features,)
            """
            self.grid.fit(train_x, train_y)


            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            # self.xgb.fit(train_x, train_y)

            """Using values instead of passing dataframe

                Returns:
                    _type_: numpy.ndarray of shape (n_features,)
            """
            self.xgb.fit(train_x.values, train_y.values)


            logging.info('XGBoost best params: ' + str(self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        
        except Exception as e:
            raise AppException(e, sys) from e


    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """
        logging.info('Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x.values) # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                logging.info('Accuracy for XGBoost:' + str(self.xgboost_score))  # Log AUC
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost) # AUC for XGBoost
                logging.info('AUC for XGBoost:' + str(self.xgboost_score)) # Log AUC

            # create best model for Random Forest
            self.random_forest=self.get_best_params_for_random_forest(train_x,train_y)
            self.prediction_random_forest=self.random_forest.predict(test_x.values) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score(test_y,self.prediction_random_forest)
                logging.info('Accuracy for RF:' + str(self.random_forest_score))
            else:
                self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest) # AUC for Random Forest
                logging.info('AUC for RF:' + str(self.random_forest_score))

            #comparing the two models
            if(self.random_forest_score <  self.xgboost_score):
                return 'XGBoost',self.xgboost
            else:
                return 'RandomForest',self.random_forest

        except Exception as e:
            raise AppException(e, sys) from e




class Kmeans_clustering:
    """
            This class shall  be used to divide the data into clusters before training.

            Written By: iNeuron Intelligence
            Version: 1.0
            Revisions: None

            """
    def __init__(self,cluster_model_path):
        self.cluster_model_path = cluster_model_path


    def elbow_plot(self,data):
        """
                        Method Name: elbow_plot
                        Description: This method saves the plot to decide the optimum number of clusters to the file.
                        Output: A picture saved to the directory
                        On Failure: Raise Exception

                        Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                """
        logging.info("Entered the elbow_plot method of the KMeansClustering class")
        wcss=[] # initializing an empty list
        try:
            for i in range (1,11):
                kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42) # initializing the KMeans object
                kmeans.fit(data) # fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
            plt.plot(range(1,11),wcss) # creating the graph between WCSS and the number of clusters
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            #plt.show()
            plt.savefig(os.path.join(self.cluster_model_path, 'K-Means_Elbow.png')) # saving the elbow plot locally
            # finding the value of the optimum cluster programmatically
            self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            logging.info('The optimum number of clusters is: '+str(self.kn.knee)+' . Exited the elbow_plot method of the KMeansClustering class')
            return self.kn.knee

        except Exception as e:
            raise AppException(e, sys) from e

    def create_clusters(self,data,number_of_clusters):
        """
                                Method Name: create_clusters
                                Description: Create a new dataframe consisting of the cluster information.
                                Output: A datframe with cluster column
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        logging.info('Entered the create_clusters method of the KMeansClustering class')
        self.data=data
        try:
            self.kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            #self.data = self.data[~self.data.isin([np.nan, np.inf, -np.inf]).any(1)]
            self.y_kmeans=self.kmeans.fit_predict(data) #  divide data into clusters

            logging.info(f"Saving clustering model to: {self.cluster_model_path}")
            save_object(file_path=os.path.join(self.cluster_model_path,'kmeans_cluster.pkl'), obj=self.kmeans)
            logging.info(f"Saved trained model to: {self.cluster_model_path}")
           

            self.data['Cluster']=self.y_kmeans  # create a new column in dataset for storing the cluster information
            logging.info('succesfully created '+str(self.kn.knee)+ 'clusters. Exited the create_clusters method of the KMeansClustering class')
            return self.data

        except Exception as e:
            raise AppException(e, sys) from e


class WaferModel:

    def __init__(self,preprocessing,clustering_obj,model_obj_list:list):
        self.preprocessing = preprocessing
        self.clustering_obj = clustering_obj
        self.model_obj_list = model_obj_list

    
    def predict(self,df):
        try:
            preprocessed_arr = self.preprocessing.transform(df)
            pred_arr=np.c_[df.values,np.array([np.NaN]*df.shape[0])]
            model_index = self.clustering_obj.predict(preprocessed_arr)
            for model_ix in range(len(self.model_obj_list)):    
                selected_record=model_index==model_ix
                pred_arr[selected_record,-1]=self.model_obj_list[model_ix].predict(preprocessed_arr[selected_record])
            column_name=df.columns.tolist()
            column_name.append('prediction')
            return pd.DataFrame(pred_arr,columns=column_name)

        except Exception as e:
            raise AppException(e, sys) from e



class ModelTrainer:

    def __init__(self, model_trainer_config, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'=' * 20}Model trainer log started.{'=' * 20} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            # self.model_list = ModelTrainer.get_list_of_models()
        except Exception as e:
            raise AppException(e, sys) from e

    
   


    def initiate_model_trainer(self):
        try:

            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            train_dataset = DataTransformation.load_numpy_array_data(
                file_path=train_file_path)
            test_dataset = DataTransformation.load_numpy_array_data(
                file_path=test_file_path)

            X_train, y_train = train_dataset[:, :-1], train_dataset[:, -1]
            # X_test, Y_test = test_dataset[:, :-1], test_dataset[:, -1]
            

            """ Applying the clustering approach"""
            clustering_model_dir = self.model_trainer_config.clustering_model_dir
            trained_model_path = self.model_trainer_config.trained_model_file_path
            os.makedirs(clustering_model_dir, exist_ok=True)
            # os.makedirs(trained_model_path, exist_ok=True)

            kmeans= Kmeans_clustering(clustering_model_dir) # object initialization.
            number_of_clusters=kmeans.elbow_plot(X_train)  #  using the elbow plot to find the number of optimum clusters
            X_train = pd.DataFrame(X_train)
            # Divide the data into clusters
            X_train =kmeans.create_clusters(X_train,number_of_clusters)

            #create a new column in the dataset consisting of the corresponding cluster assignments.
            X_train['Labels']=y_train

            # getting the unique clusters from our dataset
            list_of_clusters=X_train['Cluster'].unique()

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""
            
            list_of_model = []

            for i in list_of_clusters:
                cluster_data=X_train[X_train['Cluster']==i] # filter the data for one cluster

                # Prepare the feature and Label columns
                cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
                cluster_label= cluster_data['Labels']

                # splitting the data into training and test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)

                model_finder= Model_finder() # object initialization

                #getting the best model for each of the clusters
                best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)

                #saving the best model to the directory.
                # logging.info(f"Saving best model to: {trained_model_path}")
                list_of_model.append(best_model)
                # save_object(file_path=os.path.join(trained_model_path,best_model_name+str(i)+'.pkl'), obj=best_model)
                logging.info(f"Best model name: {best_model_name}")

            
            preprocessed_object_file_path = self.data_transformation_artifact.preprocessed_object_file_path

            preprocessed_object = load_object(file_path=preprocessed_object_file_path)
            
            cluster_object = load_object(
                file_path=os.path.join(clustering_model_dir,'kmeans_cluster.pkl'))

            trained_model = WaferModel(
                preprocessing=preprocessed_object,
                clustering_obj=cluster_object,
                model_obj_list=list_of_model)

            logging.info(f"Saving trained model to: {trained_model_path}")
            save_object(file_path=os.path.join(trained_model_path), obj=trained_model)
            logging.info(f"Saved trained model to: {trained_model_path}")


            response = ModelTrainerArtifact(is_trained=True,
                                            message="Model trained successfully",
                                            trained_model_file_path=trained_model_path,
                                            )
            logging.info(f"Trained model artifact: {response}.")
            logging.info('Successful End of Training')
            return response

        except Exception as e:
            raise AppException(e, sys) from e


    