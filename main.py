from wsgiref import simple_server
from flask import Flask, request, render_template, send_from_directory, url_for, redirect
from flask import Response
import os
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction
import pandas as pd
from flask import jsonify
from file_operations import file_methods
from application_logging import logger

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Creating a Flask App
# Flask is a Web Application Development Framework. Using this, we develop different web services or API's.
# We will be creating an API to be exposed to the client so that they will be able to leverage the functionalities that we are providing
# What functionalities?
# 1) Training and 2) Prediction

app = Flask(__name__)
dashboard.bind(app)
CORS(app)


# 3 main application routes are created
# 1) @app.route("/", methods=['GET'])
# 2) @app.route("/predict", methods=['POST'])
# 3) @app.route("/train", methods=['POST'])

# Route is a property through which one can request for the appropriate methods.


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


# @app.route("/predict", methods=['POST'])
# @cross_origin()
# def predictRouteClient():
    """
    In this predictRouteClient, there are 2 if else conditions:
    1) request.json is not None
    -> This condition is going to be used when we are testing from POSTMAN. POSTMAN is something using which we can go ahead
       and test the REST API's. So, using POSTMAN, we can test the API and when we are testing from POSTMAN, control goes to if block..

    2) request.form is not None
    -> If we are going to test from the GUI (check method home - render_template('index.html')), then the control will go to elif block..

    Note the difference in how we are obtaining the path.

    In the POSTMAN method, we are obtaining the path from request.json
    path = request.json['filepath']

    In the GUI method, we are obtaining the path from request.form
    path = request.form['filepath']

    Now, why there is this difference?
    The difference is because how our method is getting called, how this predict route is getting called from different sources.

    Whenever we are calling from UI, we are passing all the information via an HTML Form. And because of this, in this case,
    we are calling request.form method because the variables or parameters that we are passing are enclosed in a form.

    Whenever we are testing is from POSTMAN, we are passing the JSON information. And because of this, in this case, we are
    calling request.json method.


    """


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        if request.json is not None:
            path = request.json['folderPath']

            # Check if the user entered 'Prediction_SampleFile'
            if path == "Prediction_SampleFile":
                # Append the file name to the folder path
                file_path = os.path.join(path, "sample_file.csv")
            else:
                # Assume the user has provided a full file path
                file_path = path

            # Ensure the file exists before proceeding
            if not os.path.exists(file_path):
                return Response(f"Error: The file '{file_path}' does not exist.", status=400)

            # Ensure the output folder exists
            output_folder = "Prediction_Output_File"
            os.makedirs(output_folder, exist_ok=True)

            pred_val = pred_validation(path)  # object initialization

            pred_val.prediction_validation()  # calling the prediction_validation function

            pred = prediction(path)  # object initialization

            # Predicting for dataset present in database
            prediction_file_name = "Predictions.csv"
            prediction_file_path = os.path.join(
                output_folder, prediction_file_name)

            # predicting for dataset present in database
            path = pred.predictionFromModel()

            # Add your custom message here
            message = "Prediction successful! See the results below."
            predictions = path["predictions"]  # List of dictionaries

            response = {
                "message": message,
                "predictions": predictions  # This is the list of prediction rows
            }
            return jsonify(response)

            # return Response("Prediction File created at %s!!!" % path)

        elif request.form is not None:
            path = request.form['filepath']

            # Check if the user entered 'Prediction_SampleFile'
            if path == "Prediction_SampleFile":
                # Append the file name to the folder path
                file_path = os.path.join(path, "sample_file.csv")
            else:
                # Assume the user has provided a full file path
                file_path = path

            # Ensure the file exists before proceeding
            if not os.path.exists(file_path):
                return Response(f"Error: The file '{file_path}' does not exist.", status=400)

            # Ensure the output folder exists
            output_folder = "Prediction_Output_File"
            os.makedirs(output_folder, exist_ok=True)

            pred_val = pred_validation(path)  # object initialization

            # Predicting for dataset present in database
            prediction_file_name = "Predictions.csv"
            prediction_file_path = os.path.join(
                output_folder, prediction_file_name)

            pred_val.prediction_validation()  # calling the prediction_validation function

            pred = prediction(path)  # object initialization

            # predicting for dataset present in database
            path = pred.predictionFromModel()

            # Add your custom message here
            message = "Prediction successful! See the results below."
            predictions = path["predictions"]  # List of dictionaries

            response = {
                "message": message,
                "predictions": predictions  # This is the list of prediction rows
            }
            return jsonify(response)

            # return Response("Prediction File created at %s!!!" % path)

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    # in the train part, we are first doing the validation and then model training
    try:
        # client has placed the training data inside Training_Batch_Files folder
        # we will obtain that data and perform some validation on it

        # Check if 'folderPath' key exists in the request
        if 'folderPath' in request.json and request.json['folderPath'] is not None:
            path = request.json['folderPath']
            print(f"Received folderPath: {path}")  # Debug log

            # object initialization - creating 'train_valObj' object for 'train_validation' class
            train_valObj = train_validation(path)  # object initialization

            train_valObj.train_validation()  # calling the training_validation function

            trainModelObj = trainModel()  # object initialization
            trainModelObj.trainingModel()  # training the model for the files in the table

    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")


@app.route("/dynamic-predict", methods=["GET", "POST"])
@cross_origin()
def dynamic_predict():
    if request.method == "GET":
        # Serve the form for dynamic prediction
        return render_template("dynamic_predict.html")
    elif request.method == "POST":

        try:
            # Parse the incoming JSON request body
            input_data = request.get_json()

            # Validate the input data
            if not input_data:
                return jsonify({"message": "No input data provided"}), 400

            file_object = open(
                "Prediction_Logs/Dynamic_Prediction_Log.txt", "a+")
            logger_object = logger.App_Logger()

            # Log the received input data
            logger_object.log(
                file_object, f"Received input data: {input_data}")

            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])

            # Apply preprocessing to match training steps

            # Drop unnecessary columns
            columns_to_drop = ['TSH_measured', 'T3_measured',
                               'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured']
            input_df = input_df.drop(columns=columns_to_drop, errors='ignore')

            # Map binary categorical values to numeric
            binary_mapping = {'t': 1, 'f': 0}
            input_df.replace(binary_mapping, inplace=True)

            # Encode 'sex' column
            sex_mapping = {'M': 1, 'F': 0}
            if 'sex' in input_df.columns:
                input_df['sex'] = input_df['sex'].map(sex_mapping)

            # One-hot encode the referral_source column
            if 'referral_source' in input_df.columns:
                referral_source_dummies = pd.get_dummies(
                    input_df['referral_source'], prefix='referral_source')
                input_df = pd.concat(
                    [input_df, referral_source_dummies], axis=1)
                input_df.drop(['referral_source'], axis=1, inplace=True)

            # Load the KMeans model
            file_op = file_methods.File_Operation(file_object, logger_object)
            kmeans_model = file_op.load_model("KMeans")

            # Retrieve the correct feature order
            if hasattr(kmeans_model, "feature_names_in_"):
                training_feature_order = list(kmeans_model.feature_names_in_)
            else:
                raise ValueError(
                    "Feature names not found in the KMeans model.")

            # Exclude 'Class' from the feature list if present
            training_feature_order = [
                col for col in training_feature_order if col != "Class"]

            # Align input_df with the correct feature order
            for feature in training_feature_order:
                if feature not in input_df.columns:
                    # Add missing columns with default value 0
                    input_df[feature] = 0
            # Reorder columns to match training feature order
            input_df = input_df[training_feature_order]

            # Predict the cluster
            cluster_number = kmeans_model.predict(input_df)[0]
            logger_object.log(
                file_object, f"Determined cluster: {cluster_number}")

            # Load the corresponding model for the cluster
            model_name = f"KNN{cluster_number}"
            knn_model = file_op.load_model(model_name)

            # Make the prediction
            prediction = knn_model.predict(input_df)[0]

            # Ensure the prediction result is a native Python type
            prediction = int(prediction)  # Convert to native int

            # Log the prediction result
            logger_object.log(file_object, f"Prediction result: {prediction}")
            file_object.close()

            # Return the prediction result
            return jsonify({
                "message": "Prediction successful!",
                "prediction_result": prediction
            }), 200

        except Exception as e:
            if 'file_object' in locals() and not file_object.closed:
                logger_object.log(
                    file_object, f"Error during prediction: {str(e)}")
                file_object.close()
            return jsonify({"message": f"An error occurred: {str(e)}"}), 500


port = int(os.getenv("PORT", 8080))
if __name__ == "__main__":
    host = '0.0.0.0'
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
