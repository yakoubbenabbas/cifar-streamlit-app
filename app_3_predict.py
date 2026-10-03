# Streamlit Iris Prediction App: Manual Inputs + Dataset Upload — Student Exercise
#
# Run with:
#     streamlit run app_3_predict.py
#
# This final app must always use classifier.pkl and provide both prediction modes:
# one flower entered manually and many flowers supplied in an uploaded dataset.
# This exercise intentionally contains comments and TODOs only.

# TODO 1: Import pathlib.Path, pickle, NumPy as np, pandas as pd, and Streamlit as st.

# TODO 2: Load classifier.pkl relative to this script using pickle.load().

# TODO 3: Add a title and a short explanation of the two prediction modes.

# TODO 4: Create a manual-input section with four sliders using these ranges:
#         Sepal Length 4.0–8.0, Sepal Width 2.0–5.0,
#         Petal Length 1.0–7.0, and Petal Width 0.1–2.5.

# TODO 5: Add a manual "Predict" button. When clicked, build a (1, 4) NumPy
#         array, call classifier.predict(), map the class to a species name,
#         and display the result.

# TODO 6: Add an uploader accepting .csv and .xlsx files. Explain that XLSX
#         processing may require `pip install openpyxl`.

# TODO 7: Read the uploaded dataset with pandas and verify it contains:
#         sepal_length, sepal_width, petal_length, petal_width.

# TODO 8: Predict every uploaded row with classifier.predict(dataframe[...]),
#         append predicted_species, and display the completed DataFrame.

# TODO 9: Use Streamlit error messages for a missing model, unsupported file,
#         invalid columns, or malformed data.
