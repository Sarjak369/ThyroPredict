import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import pickle
from imblearn.over_sampling import RandomOverSampler


class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.

        """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_columns(self, data, columns):
        """
                Method Name: remove_columns
                Description: This method removes the given columns from a pandas dataframe.
                Output: A pandas DataFrame after removing the specified columns.
                On Failure: Raise Exception

        """
        self.logger_object.log(
            self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        self.data = data
        self.columns = columns
        try:
            # drop the labels specified in the columns
            self.useful_data = self.data.drop(labels=self.columns, axis=1)
            self.logger_object.log(self.file_object,
                                   'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.logger_object.log(
                self.file_object, 'Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,
                                   'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """
                        Method Name: separate_label_feature
                        Description: This method separates the features and a Label Coulmns.
                        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                        On Failure: Raise Exception

                """
        self.logger_object.log(
            self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            # drop the columns specified and separate the feature columns
            self.X = data.drop(labels=label_column_name, axis=1)
            self.Y = data[label_column_name]  # Filter the Label columns
            self.logger_object.log(
                self.file_object, f"Shape of X in separate_label_feature method : {self.X.shape}")
            self.logger_object.log(
                self.file_object, f"Shape of Y in separate_label_feature method : {self.Y.shape}")
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X, self.Y
        except Exception as e:
            self.logger_object.log(
                self.file_object, 'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(
                self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def dropUnnecessaryColumns(self, data, columnNameList):
        """
                        Method Name: is_null_present
                        Description: This method drops the unwanted columns as discussed in EDA section.

                                """
        data = data.drop(columnNameList, axis=1)
        return data

    def replaceInvalidValuesWithNull(self, data):
        """
                               Method Name: is_null_present
                               Description: This method replaces invalid values i.e. '?' with null, as discussed in EDA.


                                       """

        for column in data.columns:
            count = data[column][data[column] == '?'].count()
            if count != 0:
                data[column] = data[column].replace('?', np.nan)
        return data

    def is_null_present(self, data):
        """
                                Method Name: is_null_present
                                Description: This method checks whether there are null values present in the pandas Dataframe or not.
                                Output: Returns a Boolean Value. True if null values are present in the DataFrame, False if they are not present.
                                On Failure: Raise Exception


                        """
        self.logger_object.log(
            self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        try:
            # check for the count of null values per column
            self.null_counts = data.isna().sum()
            for i in self.null_counts:
                if i > 0:
                    self.null_present = True
                    break
            if (self.null_present):  # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(
                    data.isna().sum())
                # storing the null column information to file
                dataframe_with_null.to_csv(
                    'preprocessing_data/null_values.csv')
            self.logger_object.log(
                self.file_object, 'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present
        except Exception as e:
            self.logger_object.log(
                self.file_object, 'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(
                self.file_object, 'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def encodeCategoricalValues(self, data):
        """
                                           Method Name: encodeCategoricalValues
                                           Description: This method encodes all the categorical values in the training set.
                                           Output: A Dataframe which has all the categorical values encoded.
                                           On Failure: Raise Exception

                        """

        # We can map the categorical values like below:
        data['sex'] = data['sex'].map({'F': 0, 'M': 1})

        # except for 'Sex' column all the other columns with two categorical data have same value 'f' and 't'.
        # so instead of mapping indvidually, let's do a smarter work
        for column in data.columns:
            if len(data[column].unique()) == 2:
                data[column] = data[column].map({'f': 0, 't': 1})

        # this will map all the rest of the columns as we require. Now there are handful of column left with more than 2 categories.
        # we will use get_dummies with that.
        data = pd.get_dummies(data, columns=['referral_source'])

        # Label encoding is a preprocessing technique used to convert categorical data into numerical values.
        # It assigns a unique integer to each category in the column.

        """ 
        Internal Working of fit_transform:
        
        The fit method scans the column, collects all unique classes, and sorts them alphabetically:
        Classes: ['compensated_hypothyroid', 'negative', 'primary_hypothyroid', 'secondary_hypothyroid']
        Each class is assigned a unique integer based on its sorted position:
        'compensated_hypothyroid' → 0
        'negative' → 1
        'primary_hypothyroid' → 2
        'secondary_hypothyroid' → 3

        The transform method replaces the original class values with their corresponding integers.
        
        Label encoding introduces a numerical order between categories (e.g., 1 < 2 < 3). 
        For some models, this implicit ordering might lead to incorrect assumptions. 
        If the classes do not have a natural order (ordinal), one-hot encoding might be more appropriate.
        Here, label encoding is acceptable because the Class column likely represents categories without ordering concerns for the model being trained.
        
        """

        # fit: Learns the unique classes in the Class column and assigns each class a unique integer.
        encode = LabelEncoder().fit(data['Class'])

        # transform: Replaces the original categorical values in the column with the assigned integers.
        data['Class'] = encode.transform(data['Class'])

        required_columns = list(pd.get_dummies(
            data, columns=['referral_source']).columns)

        # we will save the encoder as pickle to use when we do the prediction. We will need to decode the predcited values
        # back to original
        with open('EncoderPickle/enc.pickle', 'wb') as file:
            pickle.dump(encode, file)

        return data

    def encodeCategoricalValuesPrediction(self, data):
        """
                                               Method Name: encodeCategoricalValuesPrediction
                                               Description: This method encodes all the categorical values in the prediction set.
                                               Output: A Dataframe which has all the categorical values encoded.
                                               On Failure: Raise Exception

                            """

        # We can map the categorical values like below:
        data['sex'] = data['sex'].map({'F': 0, 'M': 1})
        # we do not want to encode values with int or float type
        cat_data = data.drop(['age', 'T3', 'TT4', 'T4U', 'FTI', 'sex'], axis=1)
        # except for 'Sex' column all the other columns with two categorical data have same value 'f' and 't'.
        # so instead of mapping indvidually, let's do a smarter work
        for column in cat_data.columns:
            if (data[column].nunique()) == 1:
                # map the variables same as we did in training i.e. if only 'f' comes map as 0 as done in training
                if data[column].unique()[0] == 'f' or data[column].unique()[0] == 'F':
                    data[column] = data[column].map(
                        {data[column].unique()[0]: 0})
                else:
                    data[column] = data[column].map(
                        {data[column].unique()[0]: 1})
            elif (data[column].nunique()) == 2:
                data[column] = data[column].map({'f': 0, 't': 1})

            # we will use get dummies for 'referral_source'
        data = pd.get_dummies(data, columns=['referral_source'])

        # ---------------------------------- #

        # required_columns = [
        #     'referral_source_STMW', 'referral_source_SVHC', 'referral_source_SVHD',
        #     'referral_source_SVI', 'referral_source_other'
        # ]
        # # Add any missing columns with all-zero values
        # for col in required_columns:
        #     if col not in data.columns:
        #         data[col] = 0

        # ---------------------------------- #

        return data

    def handleImbalanceDataset(self, X, Y):
        """
                                                      Method Name: handleImbalanceDataset
                                                      Description: This method handles the imbalance in the dataset by oversampling.
                                                      Output: A Dataframe which is balanced now.
                                                      On Failure: Raise Exception

                                   """

        self.logger_object.log(
            self.file_object, 'Entered the handleImbalanceDataset method of the Preprocessor class')

        rdsmple = RandomOverSampler()
        x_sampled, y_sampled = rdsmple.fit_resample(X, Y)

        self.logger_object.log(
            self.file_object, 'Done with RandomOverSampler. Exited the handleImbalanceDataset method of the Preprocessor class')

        return x_sampled, y_sampled

    def impute_missing_values(self, data):
        """ 
                                    Method Name: impute_missing_values
                                    Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
        """

        self.logger_object.log(
            self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        try:
            self.logger_object.log(
                self.file_object, f"Shape of data before imputation: {data.shape}")

            """
            The KNNImputer imputes missing values using the K-nearest neighbors (KNN) algorithm:

            It calculates the missing values by finding the K nearest neighbors (rows with similar feature values) and using their values to estimate the missing ones.
            
            Key Parameters:
            n_neighbors: The number of nearest neighbors to consider.
            weights: Specifies how the values of neighbors contribute to the imputation:
            'uniform': All neighbors are weighted equally.
            'distance': Closer neighbors have a greater influence on the imputed value.
            missing_values: Specifies the placeholder for missing values (e.g., np.nan).


            """

            imputer = KNNImputer(
                n_neighbors=3, weights='uniform', missing_values=np.nan)
            # n_neighbors=3: The imputer will look for the 3 nearest neighbors for each missing value.
            # weights='uniform': All 3 neighbors will contribute equally when calculating the imputed value.
            # missing_values=np.nan: Indicates that missing values in the dataset are represented as NaN.

            new_array = imputer.fit_transform(data)  # Impute missing values
            """
            What Happens During fit_transform?
            The imputer learns the pattern of the data:
            It calculates the distances between rows based on their non-missing feature values.
            For rows with missing values, it identifies the K nearest neighbors based on the Euclidean distance (default).
            It fills in the missing values:
            For each missing value, the imputer calculates the mean (default for weights='uniform') or a weighted average 
            of the corresponding feature values of the K nearest neighbors.

            """

            self.logger_object.log(
                self.file_object, "KNN imputation successful")

            # Convert to DataFrame
            new_data = pd.DataFrame(data=new_array, columns=data.columns)
            """
            Why Convert to DataFrame?
            The output of fit_transform is a NumPy array, so it needs to be converted back to a DataFrame to 
            preserve the original structure and column names.
            
            """

            self.logger_object.log(
                self.file_object, f"Shape of data after imputation: {new_data.shape}")

            """
            Why Use KNNImputer?

            Preserves Relationships: KNNImputer uses neighboring rows to estimate missing values, preserving relationships between features.
            Works Well with Non-Linear Data: Unlike mean or median imputation, KNNImputer considers relationships in the data, 
            making it suitable for non-linear patterns.
            Customizable: You can adjust the number of neighbors (n_neighbors) and the weighting method to suit your dataset.


            Benefits of KNNImputer:

            Better Estimation: Compared to mean or median imputation, KNNImputer uses the structure of the data to estimate missing values.
            Handles Numerical and Categorical Data: Works well with both types of data (categorical should be encoded numerically).
            Minimizes Bias: Takes into account the relationship between features, reducing bias introduced by simple imputation techniques.
            """

            self.logger_object.log(
                self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return new_data
        except Exception as e:
            self.logger_object.log(
                self.file_object, f"Exception during imputation: {str(e)}")
        raise

    def get_columns_with_zero_std_deviation(self, data):
        """
                                                Method Name: get_columns_with_zero_std_deviation
                                                Description: This method finds out the columns which have a standard deviation of zero.
                                                Output: List of the columns with standard deviation of zero
                                                On Failure: Raise Exception

                             """
        self.logger_object.log(
            self.file_object, 'Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
        self.columns = data.columns
        self.data_n = data.describe()
        self.col_to_drop = []
        try:
            for x in self.columns:
                if (self.data_n[x]['std'] == 0):  # check if standard deviation is zero
                    # prepare the list of columns with standard deviation zero
                    self.col_to_drop.append(x)
            self.logger_object.log(
                self.file_object, 'Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            return self.col_to_drop

        except Exception as e:
            self.logger_object.log(
                self.file_object, 'Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(
                self.file_object, 'Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()
