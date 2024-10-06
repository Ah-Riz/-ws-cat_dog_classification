import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from google.cloud import firestore
import asyncio
from google.cloud import firestore
from firebase_admin import firestore

st.set_page_config(layout="wide")

load_dotenv()
db = firestore.AsyncClient(project=os.getenv("FIRESTORE_DB_PROJECT"), database=os.getenv("FIRESTORE_DB"))

async def fetch_data():
    try:
        docs = db.collection(os.getenv("FIRESTORE_DB_COLLECTION")).stream()
        res = pd.DataFrame()
        async for doc in docs:
            res = res.append(doc.to_dict(),ignore_index=True)
        return res
    except Exception as e:
        st.error(f"Failed to fetch data : {e}")
        return pd.DataFrame()
    
def filter_data(df, start_date, end_date, prediction):
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df['confidence'] = pd.to_numeric(df['confidence'])
    filtered = df[(df['time_stamp'] >= pd.to_datetime(start_date)) & (df['time_stamp'] <= pd.to_datetime(end_date))]
    if prediction != 'all':
        filtered = filtered[filtered['prediction'] == prediction]
    return filtered

def plot_confidence(df, time_frame, threshold=float(os.getenv("CONFIDENCE_THRESHOLD"))):
    if df.empty:
        st.write("No data to plot.")
        return
    
    df.set_index('time_stamp', inplace=True)
    if time_frame == 'Hourly':
        df_resample = df['confidence'].resample('H').mean()
    elif time_frame == 'Daily':
        df_resample = df['confidence'].resample('D').mean()
    elif time_frame == 'Monthly':
        df_resample = df['confidence'].resample('M').mean()
    elif time_frame == 'Yearly':
        df_resample = df['confidence'].resample('A').mean()

    plt.figure(figsize=(10, 4))
    plt.plot(df_resample.index, df_resample, marker='o', linestyle='-')
    plt.title(f'Mean Confidence Scores: {time_frame}')
    plt.ylabel('Mean Confidence')
    plt.xlabel('Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.axhline(y=threshold, color='r', linestyle='--')
    st.pyplot(plt)

async def main():
    st.title("Data Viewer for Uncertainty Table")
    try:
        data = await fetch_data()
    except Exception as e:
        st.text("error")
        st.text(e)

    default_start_date = datetime.date.today() - datetime.timedelta(days=366)

    
    with st.sidebar:

        threshold = st.number_input("Threshold", min_value=0.0, max_value=1.0, value=float(os.getenv("CONFIDENCE_THRESHOLD")), step=0.01)
        start_date = st.date_input("Start Date", value=default_start_date)
        end_date = st.date_input("End Date")

        prediction = st.selectbox("Prediction Type", ["All", "Positive", "Negative"]).lower()
        time_frame = st.selectbox("Aggregate Time Frame", ["Hourly", "Daily", "Monthly", "Yearly"])

    if not data.empty:
        data_filtered = filter_data(data, start_date, end_date, prediction)
        st.dataframe(data_filtered)
        plot_confidence(data_filtered, time_frame, threshold=threshold)
    else:
        st.write("No data available.")        

if __name__ == "__main__":
    asyncio.run(main())