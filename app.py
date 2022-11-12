import streamlit as st
import numpy as np
import pandas as pd
import pickle
import pdb

df = pd.read_csv('cleaned_house.csv')

pipe = pickle.load(open("RidgeModel.pkl","rb"))

st.title("Banglore House Price's Prediction")

location_list = df['location'].unique().tolist()

location = st.selectbox('Select Location',location_list)
sqft = st.text_input('Enter total_sqft')
bath = st.select_slider('Select Number of Bathroom',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
bhk = st.select_slider('Select BHK',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])


def predict():
	X = pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
	prediction = pipe.predict(X)[0] * 1e5
	return str(np.round(prediction))


ok = st.button("Calculate Approx. House Price")

if ok:
	predicted_amount = predict()
	st.subheader(f"The Estimated House Price is ₹{predicted_amount}")