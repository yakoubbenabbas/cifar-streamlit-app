# Streamlit Iris Dataset Prediction App — Student Exercise
#
# Run with:
#     streamlit run app_2_uploader.py
#
# This app must always load classifier.pkl and use it to generate predictions
# for every row in an uploaded CSV or XLSX dataset.
# This exercise intentionally contains comments and TODOs only.

# TODO 1: Import pathlib.Path, pickle, pandas as pd, and Streamlit as st.

# TODO 2: Build a path to classifier.pkl relative to this script, open it in
#         binary read mode, and load the model with pickle.load().
from pathlib import Path
import pickle
import numpy as np
import streamlit as st
import pandas as pd


# TODO 2: Build the path to classifier.pkl relative to this Python file.
#         Open the file in binary read mode and load it with pickle.load().
#         Store the trained model in a variable called `classifier`.

model_path = Path(__file__).parent / "model.pkl"

with open(model_path, "rb") as model:
    classifier = pickle.load(model)

# TODO 3: Add a title explaining that the app performs batch Iris predictions.
st.title("iris batch prediction")

# TODO 4: Create st.file_uploader() accepting .csv and .xlsx files.
#         Remind users that Excel support may require: pip install openpyxl
uploaded_file=st.file_uploader("upload csv or xlsx file", type=["csv","xlsx"])

# TODO 5: If a file was uploaded, identify its extension and read it with:
#         - pd.read_csv(uploaded_file) for CSV files
#         - pd.read_excel(uploaded_file, engine="openpyxl") for XLSX files

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(",")[1]
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_extension =="xlsx":
        df = pd.read_excel(uploaded_file)
    else:
        st.info("upload csv or excel file to proceed.")

# TODO 6: Define the four required feature columns in training order:
#         sepal_length, sepal_width, petal_length, petal_width.
#         Check that the uploaded DataFrame contains all four columns.
    feature_columns = ["sepal_length,sepal_width,petal_length,petal_width"]

    if all(col in df.columns for col in feature_columns):

        pass
    else:
        st.error(f"upload file with columns:{','.join(feature_columns)}")

# TODO 7: Pass dataframe[feature_columns] directly to classifier.predict().
    predictions = classifier.predict(df[feature_columns])
    species = ["setosa","versicolor","virginica"]

    final_df = df.copy()
    final_df["predicted_species"] = [species[pred] for pred in predictions]

# TODO 8: Add a `predicted_species` column using the class-name order:
#         ["setosa", "versicolor", "virginica"].

# TODO 9: Display the input data and predictions with st.dataframe().
#         Include a helpful error message when columns or file type are invalid.

    st.dataframe(final_df)
else:
    st.info("upload a csv or xlsx")