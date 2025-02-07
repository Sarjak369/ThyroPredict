# ThyroPredict
ThyroPredict: Thyroid Disorder Classification for Medical Diagnostics

**ThyroPredict** is a machine learning-based system designed to predict thyroid disorders with **up to 92% accuracy** and **98% ROC AUC scores**. Built with Python, Flask, and Scikit-learn, this project leverages advanced data preprocessing, clustering, and model training techniques to provide accurate predictions for thyroid conditions. The system is hosted on **AWS EC2** for real-time and bulk predictions, enabling healthcare teams to make informed decisions quickly.

---
<img width="1211" alt="Screenshot 2025-02-06 at 8 34 42 PM" src="https://github.com/user-attachments/assets/33d051a1-087d-4aa0-9b63-4e1757f044ba" />

<img width="1233" alt="Screenshot 2025-02-06 at 8 34 53 PM" src="https://github.com/user-attachments/assets/9fcddd7a-29d3-4f18-889d-ad8fc3878fa7" />

<img width="940" alt="Screenshot 2025-02-06 at 8 39 55 PM" src="https://github.com/user-attachments/assets/dee1288f-7840-4890-8430-1e26cbebc8af" />

<img width="1278" alt="Screenshot 2025-02-06 at 8 40 32 PM" src="https://github.com/user-attachments/assets/00a8c619-4e9a-45b0-9a9b-30ae7685fa92" />

---

## 🚀 Key Features
- **High Accuracy**: Achieved **92% accuracy** and **98% ROC AUC** using KNeighborsClassifier and Random Forest.
- **Dynamic Data Segmentation**: Utilized **KMeans clustering** to segment data into clusters, improving model performance.
- **Class Imbalance Handling**: Addressed class imbalance using **RandomOverSampler**.
- **Real-Time Predictions**: Hosted on **AWS EC2**, enabling real-time predictions via a Flask web app.
- **Automated Data Validation**: Ensures data integrity with automated validation and preprocessing pipelines.
- **Bulk Predictions**: Supports bulk predictions with CSV export for healthcare teams.

---

## 📊 Problem Statement
Thyroid disease is a common medical condition that is difficult to diagnose early. This project aims to build a classification system to predict whether a patient is suffering from thyroid disease and, if so, classify the type of thyroid disorder. The system uses patient data to provide early detection and accurate identification, aiding doctors in making better treatment decisions.

### Target Classes:
- **Compensated Hypothyroid**
- **Negative** (No thyroid disorder)
- **Primary Hypothyroid**
- **Secondary Hypothyroid**

---

## 📂 Dataset

<img width="1042" alt="Screenshot 2025-02-06 at 9 06 16 PM" src="https://github.com/user-attachments/assets/53695f96-4b1a-4348-8597-3c30f60ef12f" />


The dataset consists of **3772 rows** and **30 columns**, including features such as:
- **Age**: Age of the patient.
- **Sex**: Gender of the patient (M/F).
- **On Thyroxine**: Whether the patient is on thyroxine treatment.
- **TSH**: Thyroid Stimulating Hormone levels.
- **T3/T4**: Thyroid hormone levels.
- **Class**: Target variable indicating the type of thyroid disorder.

---

## 🛠️ Architecture

<img width="634" alt="Screenshot 2025-02-06 at 9 05 38 PM" src="https://github.com/user-attachments/assets/53d46754-0702-4719-8c3c-4d9e3c6eeb88" />


The project follows a modular architecture, breaking down the problem into smaller components for better maintainability and collaboration. The key components include:

### 1. **Data Ingestion Pipeline**
- **Data Validation**: Validates file names, column names, data types, and null values.
- **Data Transformation**: Converts categorical values and handles missing data.
- **Data Insertion**: Aggregates and stores validated data in a database.

### 2. **Model Training Pipeline**
- **Data Preprocessing**: Handles missing values, encodes categorical features, and balances the dataset using **RandomOverSampler**.
- **Clustering**: Uses **KMeans clustering** to segment data into clusters for better model performance.
- **Model Selection**: Trains separate models for each cluster using **Random Forest** and **KNN**, selecting the best model based on AUC scores.
- **Hyperparameter Tuning**: Optimizes model parameters for each cluster.

### 3. **Prediction Pipeline**
- **Data Validation**: Validates prediction data similar to the training pipeline.
- **Clustering**: Predicts the cluster for new data using the saved KMeans model.
- **Prediction**: Uses the best model for each cluster to make predictions.
- **CSV Export**: Exports predictions for healthcare teams.

### 4. **Deployment**
- **AWS EC2**: Hosted on an EC2 instance for real-time predictions.
- **Flask Web App**: Provides a user-friendly interface for predictions.

---

## 📈 Results
### Model Performance
| Algorithm           | Accuracy | ROC AUC |
|---------------------|----------|---------|
| KNeighborsClassifier| 92%      | 98%     |
| Random Forest       | 90%      | 97%     |

### Key Insights
- **Dynamic Clustering**: Data segmentation into clusters improved model accuracy.
- **Class Imbalance Handling**: RandomOverSampler effectively balanced the dataset.
- **Real-Time Predictions**: The Flask app enables quick and accurate predictions.

---


## Logs

### Training Log

<img width="1357" alt="Screenshot 2025-02-06 at 8 42 28 PM" src="https://github.com/user-attachments/assets/def73432-68bf-4756-86b3-3798856a54b9" />


### Prediction Log

<img width="1258" alt="Screenshot 2025-02-06 at 8 42 58 PM" src="https://github.com/user-attachments/assets/d4b31cdf-8c4a-4ba2-8cb4-2ee435f3a055" />


--- 


## Code Structure

- Data Ingestion
- Data Preprocessing
- Model Selection
- Model Tuning
- Prediction
- Logging Framework
- Deployment
- Model Retraining

---

## 🛠️ Installation

Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ThyroPredict.git
   cd ThyroPredict
   ```



## Empowering healthcare with AI-driven diagnostics! ✨




