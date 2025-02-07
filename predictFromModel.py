import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
import pickle


class prediction:

    def __init__(self, path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):

        try:
            # deletes the existing prediction file from last run!
            self.pred_data_val.deletePredictionFile()
            self.log_writer.log(self.file_object, 'Start of Prediction')
            data_getter = data_loader_prediction.Data_Getter_Pred(
                self.file_object, self.log_writer)
            data = data_getter.get_data()

            preprocessor = preprocessing.Preprocessor(
                self.file_object, self.log_writer)
            data = preprocessor.dropUnnecessaryColumns(data,
                                                       ['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured',
                                                        'FTI_measured', 'TBG_measured', 'TBG', 'TSH'])

            # replacing '?' values with np.nan as discussed in the EDA part

            data = preprocessor.replaceInvalidValuesWithNull(data)

            # get encoded values for categorical data

            data = preprocessor.encodeCategoricalValuesPrediction(data)
            is_null_present = preprocessor.is_null_present(data)
            if (is_null_present):
                data = preprocessor.impute_missing_values(data)

            # data=data.to_numpy()
            file_loader = file_methods.File_Operation(
                self.file_object, self.log_writer)
            kmeans = file_loader.load_model('KMeans')

            clusters = kmeans.predict(data)
            data['clusters'] = clusters
            clusters = data['clusters'].unique()
            result = []  # initialize blank list for storing predicitons
            # let's load the encoder pickle file to decode the values
            with open('EncoderPickle/enc.pickle', 'rb') as file:
                encoder = pickle.load(file)

            for i in clusters:
                cluster_data = data[data['clusters'] == i]
                cluster_data = cluster_data.drop(['clusters'], axis=1)
                model_name = file_loader.find_correct_model_file(i)
                model = file_loader.load_model(model_name)
                for val in (encoder.inverse_transform(model.predict(cluster_data))):
                    result.append(val)
            result = pandas.DataFrame(result, columns=['Predictions'])
            path = "Prediction_Output_File/Predictions.csv"
            # appends result to prediction file
            result.to_csv(
                "Prediction_Output_File/Predictions.csv", header=True)
            self.log_writer.log(self.file_object, 'End of Prediction')

            # Return predictions as JSON for UI display
            return {
                "message": f"Prediction File created at {path}!!!",
                "predictions": result.to_dict(orient="records")
            }

        except Exception as ex:
            self.log_writer.log(
                self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path
