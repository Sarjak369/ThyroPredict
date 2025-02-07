"""
This is the Entry point for Training the Machine Learning Model.

"""


# Doing the necessary imports
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger
import pickle

# Creating the common Logging object


class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from the source
            data_getter = data_loader.Data_Getter(
                self.file_object, self.log_writer)
            data = data_getter.get_data()

            """doing the data preprocessing"""

            preprocessor = preprocessing.Preprocessor(
                self.file_object, self.log_writer)

            # removing unwanted columns as discussed in the EDA part in ipynb file
            data = preprocessor.dropUnnecessaryColumns(
                data, ['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 'TBG', 'TSH'])

            # repalcing '?' values with np.nan as discussed in the EDA part

            data = preprocessor.replaceInvalidValuesWithNull(data)

            # get encoded values for categorical data

            data = preprocessor.encodeCategoricalValues(data)

            # create separate features and labels
            X, Y = preprocessor.separate_label_feature(
                data, label_column_name='Class')

            # check if missing values are present in the dataset
            is_null_present = preprocessor.is_null_present(X)

            # if missing values are there, replace them appropriately.
            if (is_null_present):
                X = preprocessor.impute_missing_values(
                    X)  # missing value imputation

            """
            What is RandomOverSampler?
            RandomOverSampler is a class from the imbalanced-learn library (imblearn) that addresses the problem of class imbalance in datasets. 
            It is a resampling technique that balances the dataset by increasing the number of samples in the minority class(es) through random oversampling. 
            This is done by duplicating existing minority class samples or creating synthetic samples for those classes.

            Why are we using RandomOverSampler?
            Class imbalance is a common issue in datasets where one or more classes have significantly fewer samples compared to others. 
            This imbalance can lead to biased machine learning models that prioritize the majority class and perform poorly on the minority class.


            """
            X, Y = preprocessor.handleImbalanceDataset(X, Y)

            # Applying the clustering approach

            """
            Why Clustering?
            If we go ahead and do a clustering and then try to fit individual models to those clusters, then we will be ending up with a model which performs better.
            That is why we are going to follow clustering approach.
            
            Which Clustering?
            Here, we are going to go ahead with the K-Means Clustering. With K-Means Clustering we have a prerequisite that we need to specify the value of "k".
            What is the value of k? In order to calculate the value of k, we can go ahead with the elbow method, or we can go ahead with a library called as "kneed".
            Using kneed library, programatically, we can find the value of the elbow/knee and that will give us an appropriate cluster number. 

            """

            # object initialization.
            kmeans = clustering.KMeansClustering(
                self.file_object, self.log_writer)
            # using the elbow plot to find the number of optimum clusters
            number_of_clusters = kmeans.elbow_plot(X)

            """
            
            Elbow Method: The optimal number of clusters will be determined using the Elbow Method. (We got 3)
            The Elbow Method calculates the Within-Cluster-Sum of Squared Errors (WCSS) for various cluster counts.
            WCSS decreases as more clusters are added. 
            The "elbow point" is where the decrease slows, indicating the optimal number of clusters.

            KMeans Clustering: Data was grouped into 3 clusters, reducing intra-cluster variance.
            
            """

            self.log_writer.log(
                self.file_object, 'Printing number_of_clusters for debugging:{}'.format(number_of_clusters))

            # Divide the data into clusters
            X = kmeans.create_clusters(X, number_of_clusters)

            # create a new column in the dataset consisting of the corresponding cluster assignments.
            X['Labels'] = Y

            # getting the unique clusters from our dataset
            list_of_clusters = X['Cluster'].unique()
            self.log_writer.log(
                self.file_object, 'Printing list_of_clusters for debugging:{}'.format(list_of_clusters))

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            for i in list_of_clusters:
                # filter the data for one cluster
                cluster_data = X[X['Cluster'] == i]

                # Prepare the feature and Label columns
                cluster_features = cluster_data.drop(
                    ['Labels', 'Cluster'], axis=1)  # dropping cluster and labels columns
                cluster_label = cluster_data['Labels']

                # splitting the data into training and test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(
                    cluster_features, cluster_label, test_size=1 / 3, random_state=355)

                model_finder = tuner.Model_Finder(
                    self.file_object, self.log_writer)  # object initialization

                # Save feature order for later use in predictions
                with open('feature_order.pkl', 'wb') as f:
                    pickle.dump(list(X.columns), f)

                # getting the best model for each of the clusters
                best_model_name, best_model = model_finder.get_best_model(
                    x_train, y_train, x_test, y_test)

                # saving the best model to the directory.
                file_op = file_methods.File_Operation(
                    self.file_object, self.log_writer)
                save_model = file_op.save_model(
                    best_model, best_model_name+str(i))

            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception:
            # logging the unsuccessful Training
            self.log_writer.log(
                self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception
