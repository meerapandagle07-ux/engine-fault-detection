
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_excel("engine_fault_dataset_10000.xlsx")

# Features & Target
X = data[['Temperature', 'Vibration', 'Sound']]
y = data['Condition']

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# UI
st.title("🚗 Engine Fault Detection System")

st.write("Enter engine details:")

temp = st.slider("Temperature", 60, 100)
vib = st.selectbox("Vibration", [0,1,2])
sound = st.selectbox("Sound", [0,1,2])

if st.button("Predict"):
    result = model.predict([[temp, vib, sound]])
    st.success("Condition: " + result[0])
