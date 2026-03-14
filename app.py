import streamlit as st
import pickle
import pandas as pd

st.title("🏏 IPL Win Probability Predictor")

pipe = pickle.load(open("models/pipe.pkl","rb"))

teams = [
'Chennai Super Kings','Mumbai Indians','Royal Challengers Bengaluru',
'Kolkata Knight Riders','Punjab Kings','Rajasthan Royals',
'Delhi Capitals','Sunrisers Hyderabad'
]

cities = [
'Mumbai','Delhi','Bangalore','Kolkata','Chennai','Hyderabad','Jaipur','Mohali'
]

batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", teams)
city = st.selectbox("City", cities)

runs_left = st.number_input("Runs Left")
balls_left = st.number_input("Balls Left")
wickets = st.number_input("Wickets Remaining")

target = st.number_input("Target")

if st.button("Predict Probability"):

    crr = (target - runs_left) / (120 - balls_left)
    rrr = runs_left / balls_left * 6

    input_df = pd.DataFrame({
        'batting_team':[batting_team],
        'bowling_team':[bowling_team],
        'city':[city],
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wickets_remaining':[wickets],
        'runs_target':[target],
        'crr':[crr],
        'rrr':[rrr]
    })

    result = pipe.predict_proba(input_df)

    st.subheader("Win Probability")
    st.write(batting_team, result[0][1])
    st.write(bowling_team, result[0][0])
