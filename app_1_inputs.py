# Streamlit Iris Prediction App: Manual Inputs — Student Exercise
#
# Run with:
#     streamlit run app_1_inputs.py
#
# This app must always use the saved classifier.pkl model. Place classifier.pkl
# beside this script before launching Streamlit.
# This exercise intentionally contains comments and TODOs only.

# TODO 1: Import pathlib.Path, pickle, NumPy as np, and Streamlit as st.
from pathlib import Path
import pickle
import numpy as np
import streamlit as st


# TODO 2: Build the path to classifier.pkl relative to this Python file.
#         Open the file in binary read mode and load it with pickle.load().
#         Store the trained model in a variable called `classifier`.

model_path = Path(__file__).parent / "model.pkl"

with open(model_path, "rb") as model:
    classifier = pickle.load(model)

# TODO 3: Add a title such as "Iris Manual Prediction".

st.title("Iris Manual Prediction")

# TODO 4: Create four st.slider() widgets for the Iris features:
#         - Sepal Length: 4.0 to 8.0
#         - Sepal Width: 2.0 to 5.0
#         - Petal Length: 1.0 to 7.0
#         - Petal Width: 0.1 to 2.5
#         Give each slider a useful default value and a step of 0.1.

sepal_length = st.slider("sepal length cm", min_value=4.0, max_value=8.0, value=5.0, step=0.1)
sepal_width = st.slider("sepal width cm", min_value=2.0, max_value=5.0, value=3.0, step=0.1)
petal_length = st.slider("petal length cm", min_value=1.0, max_value=7.0, value=4.0, step=0.1)
petal_width = st.slider("petal lwidth cm", min_value=0.1, max_value=2.5, value=1.0, step=0.1)

# TODO 5: Create a button named "Predict" or "Classify".

predict_button  = st.button("predict")
# TODO 6: When the button is clicked, place the four slider values into one
#         2D NumPy array with shape (1, 4), in the same order used for training.
if predict_button:
    input_features = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
# TODO 7: Call classifier.predict(input_features) and convert the numeric
#         output to one of: setosa, versicolor, or virginica.
    prediction = classifier.predict(input_features)
    predicted_species = ["setosa","versicolor","virginica"][prediction[0]]

# TODO 8: Display the predicted species clearly with st.success() or st.write().
    st.success(f"prediction {predicted_species}")