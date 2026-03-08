# Neural Network Admission Predictor

A modular machine learning project built for **CST2216 – Modularizing and Deploying ML Code** at **Algonquin College**.

This application predicts whether a student is likely to fall into the **high-admission category** using a neural network classification model built with **scikit-learn's MLPClassifier** and deployed with **Streamlit**.

## Project Overview

This project was developed by **Mohammed Laalahmi** under the supervision of **Professor Dr. Umer Altaf** as part of the Business Intelligence Systems Infrastructure program at Algonquin College.

The original notebook solution was transformed into a complete modular ML application with:

- modular Python source code
- logging and error handling
- training pipeline
- saved model artifacts
- interactive Streamlit interface
- GitHub version control
- deployment-ready project structure

## Dataset

The project uses the UCLA admissions dataset stored as:

`data/Admission.csv`

### Input Features

- GRE_Score
- TOEFL_Score
- University_Rating
- SOP
- LOR
- CGPA
- Research

### Target

- `Admit_Chance`

The original continuous target was converted into a binary class using this rule:

- `Admit_Chance >= 0.80` → `1` (Likely Admit)
- `Admit_Chance < 0.80` → `0` (Unlikely Admit)

## Project Structure

```text
neural-network-admission-ml/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── assets/
│   └── algonquin_logo.png
│
├── artifacts/
│   └── model_bundle.joblib
│
├── data/
│   └── Admission.csv
│
├── logs/
│
├── notebooks/
│   └── UCLA_Neural_Networks_Solution.ipynb
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── logger.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── model.py
    ├── evaluate.py
    ├── train.py
    ├── predict.py
Technologies Used

Python

pandas

numpy

scikit-learn

joblib

Streamlit

Model Workflow

The project follows this workflow:

Load and validate the dataset

Prepare features and binary target

Split data into training and testing sets

Scale features using MinMaxScaler

Train multiple MLPClassifier models

Compare models using classification metrics

Select the best model based on F1-score

Save the best model bundle with preprocessing objects

Serve predictions through a Streamlit application

Candidate Models

The following neural network models were compared:

mlp_small_relu

mlp_medium_relu

mlp_medium_tanh

The best model is automatically selected during training and saved in:

artifacts/model_bundle.joblib

Evaluation Metrics

The models are evaluated using:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

Classification Report

How to Run the Project Locally
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/neural-network-admission-ml.git
cd neural-network-admission-ml
2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run model training
python -m src.train
5. Launch the Streamlit app
streamlit run app.py
Streamlit App Features

modern and attractive user interface

applicant profile form

prediction label

admission probability

model confidence

trained model metrics in sidebar

Algonquin College logo

academic credit section

Deployment

This project is designed for deployment on Streamlit Cloud.

To deploy:

Push the project to GitHub

Sign in to Streamlit Cloud

Select the repository

Set app.py as the main file

Deploy

Author and Academic Credit

Developed by: Mohammed Laalahmi
Program: Business Intelligence Systems Infrastructure
Institution: Algonquin College
Professor: Dr. Umer Altaf

License

This project is for academic and educational purposes.