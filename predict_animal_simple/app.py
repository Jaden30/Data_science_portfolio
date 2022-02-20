import streamlit as st 
import pandas as pd 
import joblib


#Title 
st.header("Predicting the animal type")

#input bar 1 
height = st.number_input("Enter height")

#input bar 2
weight = st.number_input("Enter weight")

# Drop down select for eye color 
eyes = st.selectbox("Select Eye Colour", ("Blue", "Brown"))

# if button is pressed 
if st.button("SUBMIT"): 
    # UNPICKLE CLASSIFIER 
    clf = joblib.load("clf.pkl")
    
    #Store inputs into dataframe 
    X = pd.DataFrame([[height, weight, eyes]],
                    columns = ["Height", "Weight", "Eyes"])
    X = X.replace(["Brown", "Blue"], [1,0])
    
    #Get prediction 
    prediction = clf.predict(X)[0]
    
    # Output prediction 
    st.text(f"This instance is a {prediction}")