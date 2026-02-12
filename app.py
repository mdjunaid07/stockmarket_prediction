# app.py
import streamlit as st
import pandas as pd
import numpy as np
from utils import download_data, prepare_features, build_and_train_model, save_model, load_model, iterative_predict

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("Stock Market Prediction — Streamlit")

with st.sidebar:
    ticker = st.text_input("Ticker (Yahoo)", value="AAPL")
    start_date = st.date_input("Start date", value=pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End date", value=pd.to_datetime("today"))
    days_ahead = st.slider("Days to predict ahead (business days)", 1, 30, 7)
    run_btn = st.button("Run")

if run_btn:
    try:
        df = download_data(ticker, start=start_date, end=end_date)
        if df.empty:
            st.warning("No data for this ticker / date range.")
        else:
            st.subheader(f"Historical data for {ticker}")
            st.dataframe(df.tail(10))

            st.subheader("Close price chart")
            st.line_chart(df['Close'])

            # Prepare features and training
            X, y, feature_cols, df_features = prepare_features(df, days_ahead=days_ahead)

            if len(X) < 50:
                st.warning("Not enough data rows after feature creation. Try earlier start date.")
            else:
                # split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = build_and_train_model(X_train, y_train)

                # Evaluate
                from sklearn.metrics import mean_squared_error
                y_pred = model.predict(X_test)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                st.write(f"Test RMSE: {rmse:.4f}")

                # Save model
                save_model(model)

                # Iterative forecast
                last_row = df_features.iloc[-1]
                preds = iterative_predict(last_row, model, feature_cols, days_ahead=days_ahead)

                # Generate dates (business days)
                future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days_ahead)
                pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': preds}).set_index('Date')

                st.subheader("Predictions")
                st.dataframe(pred_df)
                st.line_chart(pred_df)

                # allow CSV download
                csv = pred_df.to_csv().encode('utf-8')
                st.download_button("Download predictions CSV", csv, file_name=f"{ticker}_predictions.csv", mime="text/csv")





                

    except Exception as e:
        st.error(f"Error: {e}")




# ====== SIMPLE FUTURE PRICE PREDICTION (NO DATES) ======

# Iterative prediction
last_row = df_features.iloc[-1]
preds = iterative_predict(last_row, model, feature_cols, days_ahead=days_ahead)

# Create a simple table with only predicted prices
future_days = [f"Day {i+1}" for i in range(days_ahead)]
pred_df = pd.DataFrame({
    "Future Day": future_days,
    "Predicted Close Price": preds
})

st.subheader("Future Predicted Prices")
st.dataframe(pred_df)

# Line chart without dates (using index)
pred_chart_df = pd.DataFrame(preds, columns=["Predicted Price"])
st.line_chart(pred_chart_df)
