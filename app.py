import streamlit as st
import pandas as pd
import pickle

st.title("Health Insurance Cost Prediction ")

with open("insurance_pipeline.pkl", 'rb') as f:
    pipeline = pickle.load(f)

expected_features = list(pipeline.feature_names_in_)

# Input data
age = st.slider("Select your age: ", 18,100)
sex = st.selectbox("Select your gender: ", ["male", "female"])
bmi = st.number_input("Enter your BMI: ",10.0, 50.0,25.0)
children = st.slider("Number of childrens",0,10,0)
smoker = st.selectbox("Do You Smoke ?", ["no","yes"])
region = st.selectbox("Select your region: ",["northeast","northwest","southeast","southwest"])

#convert categorical into numeric
sex = 0 if sex == "female" else 1
smoker = 0 if smoker == "no" else 1

region_northwest = 1 if region == "northwest" else 0
region_southwest = 1 if region == "southwest" else 0
region_southeast = 1 if region == "southeast" else 0

# Create a dataframe just like the model expects

input_data = pd.DataFrame({
    'age':[age],
    'sex':[sex],
    'bmi':[bmi],
    'children':[children],
    'smoker':[smoker],
    'region_northwest':[region_northwest],
    'region_southeast':[region_southeast],
    'region_southwest':[region_southwest],
})

input_data = input_data.reindex(columns=expected_features)

st.subheader("Your Input Data")
st.write(input_data)

if st.button("Predict My Insurance Cost 💸"):
    prediction = pipeline.predict(input_data)[0]
    st.success(f"Estimated Insurance Cost: ${prediction:.2f}")

    if smoker == 1:
        st.warning("Smoking significantly increases insurance costs. Consider quitting for better health and lower premiums.")
    else:
        st.info("Great job on not smoking! This helps keep your insurance costs lower and benefits your health.")
    

