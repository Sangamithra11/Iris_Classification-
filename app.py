import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('iris_model.pkl')

st.title("Iris Flower Classification")

sepal_length = st.number_input("Sepal Length (cm)", 0.0,10.0, 5.1, step=0.1)
sepal_width  = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5, step=0.1)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4, step=0.1)
petal_width  = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2, step=0.1)

input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

if st.button("Predict Species"):
    prediction = model.predict(input_data)
    st.success(f"The predicted species is **{prediction[0]}**")
