
import streamlit as st
import pandas as pd
import openai
from datetime import datetime, timedelta

# Load OpenAI key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load CSV
@st.cache_data
def load_data():
    df = pd.read_csv("sales_history.csv")
    df['Start_Date'] = pd.to_datetime(df['Start_Date'])
    return df.sort_values(by=['Brand', 'Start_Date'])

# Predict next sale
def get_prediction(brand, df):
    brand_df = df[df['Brand'].str.lower() == brand.lower()]
    if len(brand_df) < 2:
        return f"Not enough data to predict sales for {brand.title()}.", None

    dates = brand_df['Start_Date'].tolist()
    intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    avg_days = round(sum(intervals) / len(intervals))
    next_sale = dates[-1] + timedelta(days=avg_days)

    context = (f"{brand.title()} has had sales on these dates: " +
               ", ".join(d.strftime('%b %d, %Y') for d in dates) +
               f". The average time between sales is {avg_days} days.")
    
    return context, next_sale.strftime('%b %d, %Y')

# App UI
st.title("🛍️ GPT Sale Predictor")
st.write("Ask about a brand's next likely sale:")

user_input = st.text_input("Try asking: 'When is the next Nike sale?'")

if user_input:
    df = load_data()
    # Extract brand name from question
    brand_guess = user_input.split()[-2]  # crude but works for demo
    context, prediction = get_prediction(brand_guess, df)

    if prediction:
        prompt = f"{context} Based on that, predict when their next sale will be. Respond as a helpful assistant."

        with st.spinner("Thinking..."):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful shopping assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.write(f"🧠 GPT says:

{response['choices'][0]['message']['content']}")
    else:
        st.write(context)
