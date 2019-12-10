from sklearn.externals import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title('Phone price')
st.write('adfajf')
data = pd.read_csv('Advertising.csv')

data 
# draws a histogram
st.subheader('Advertising Costs')

# use numpy to generate bins for age
hist_values = np.histogram(data.TV, bins=300 , range=(0,300))

# Show Bar Chart
st.bar_chart(hist_values)


#slider 1
TV = st.slider('TV Advertising Cost', 0 , 300, 150 )

radio = st.slider('Radio Advertising Cost', 0 , 50, 25)

newspaper = st.slider('Newspaper Advertising Cost', 0, 250, 125)

# title
st.subheader('Predicted Sales')

#load saved machine learning model
filename = 'model.sav'
saved_model  = joblib.load(filename)

#predict sales using variables
predicted_sales = saved_model.predict([[TV, radio]])[0]

#print prediction
st.write(predicted_sales)