import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("disease_model.pkl", "rb"))

train = pd.read_csv("Training.csv")
train = train.loc[:, ~train.columns.str.contains('^Unnamed')]
symptoms_list = train.drop("prognosis", axis=1).columns.tolist()

st.title("SMART COMMUNITY HEALTH MONITORING SYSTEM")

selected_symptoms = st.multiselect("Select Symptoms:", symptoms_list)

if st.button("Predict"):
    input_data = [0] * len(symptoms_list)
    
    for symptom in selected_symptoms:
        index = symptoms_list.index(symptom)
        input_data[index] = 1
    
    prediction = model.predict([input_data])
    
    st.success(f"Predicted Disease: {prediction[0]}")