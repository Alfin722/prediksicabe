import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Judul Aplikasi
st.title("Prediksi Harga Komoditas Cabai dengan LSTM")
st.markdown(
    "Aplikasi ini memprediksi harga cabai per bulan menggunakan model LSTM yang telah dilatih."
)

@st.cache_resource
def load_lstm_model(model_path='lstm_model.h5'):
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

@st.cache_resource
def load_scaler(scaler_bytes=None):
    if scaler_bytes:
        try:
            return pickle.load(scaler_bytes)
        except Exception as e:
            st.error(f"Gagal memuat scaler dari file: {e}")
            return None
    else:
        try:
            with open('scaler.pkl', 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

model = load_lstm_model()

st.sidebar.header("Input File")
uploaded_file = st.sidebar.file_uploader(
    "Upload file harga cabai (CSV atau XLSX) dengan kolom 'Date' dan 'Price'", 
    type=["csv", "xlsx"]
)
scaler_file = st.sidebar.file_uploader(
    "Upload file scaler.pkl", type=["pkl", "pickle"]
)

scaler = load_scaler(scaler_file) if scaler_file else load_scaler()

def load_dataframe(uploaded):
    name = uploaded.name.lower()
    if name.endswith('.xlsx'):
        return pd.read_excel(uploaded, engine='openpyxl', parse_dates=['Date'])
    return pd.read_csv(uploaded, parse_dates=['Date'])

if uploaded_file and model:
    # Load dan set index tanggal
    df = load_dataframe(uploaded_file).sort_values('Date')
    df.set_index('Date', inplace=True)

    # Agregasi data harian ke bulanan (rata-rata)
    df_monthly = df['Price'].resample('M').mean().to_frame()
    df_monthly.index = df_monthly.index.to_period('M').to_timestamp('M')

    st.subheader("Data Historis Harga Cabai per Bulan")
    st.line_chart(df_monthly['Price'])

    prices = df_monthly['Price'].values.reshape(-1, 1)

    # Tentukan panjang sequence bulanan
    max_seq = len(df_monthly)
    if max_seq > 1:
        default_seq = min(12, max_seq-1)
        seq_max = max_seq-1
    else:
        default_seq = 1
        seq_max = 1

    sequence_length = st.sidebar.number_input(
        'Panjang sequence (jumlah bulan untuk input LSTM)',
        value=default_seq, min_value=1, max_value=seq_max
    )

    # Scaling
    if scaler:
        scaled = scaler.transform(prices)
    else:
        temp_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = temp_scaler.fit_transform(prices)
        scaler = temp_scaler

    # Prediksi periode berikutnya
    last_seq = scaled[-sequence_length:]
    X_input = np.expand_dims(last_seq, axis=0)
    pred_scaled = model.predict(X_input)
    pred_price = scaler.inverse_transform(pred_scaled)[0, 0]

    st.subheader("Hasil Prediksi Bulanan")
    st.markdown(f"**Prediksi harga cabai untuk bulan berikutnya:** {pred_price:.2f}")
    st.table(pd.DataFrame({'Type': ['Terakhir (Real)', 'Prediksi'], 'Price': [prices[-1,0], pred_price]}))

    # Evaluasi rolling
    if len(scaled) > sequence_length:
        preds = []
        for i in range(sequence_length, len(scaled)):
            seq_input = np.expand_dims(scaled[i-sequence_length:i], axis=0)
            pred_i = model.predict(seq_input)[0,0]
            preds.append(pred_i)
        preds = np.array(preds).reshape(-1,1)

        pred_prices = scaler.inverse_transform(preds).flatten()
        true_prices = prices[sequence_length:].flatten()

        mape_series = np.abs((true_prices - pred_prices) / true_prices) * 100
        overall_mape = np.mean(mape_series)

        st.subheader(f"MAPE {overall_mape:.2f}%")

        st.subheader("Perbandingan Grafik Real vs Prediksi")
        comparison_df = pd.DataFrame(
            {'Real': true_prices, 'Prediksi': pred_prices},
            index=df_monthly.index[sequence_length:]
        )
        st.line_chart(comparison_df)
    else:
        st.info("Data historis bulanan terlalu sedikit untuk evaluasi rolling.")

else:
    if not model:
        st.error("Model LSTM tidak tersedia. Pastikan 'lstm_model.h5' tersedia di direktori.")
    else:
        st.info("Silakan upload data historis harga cabai di sidebar.")

# Footer
st.markdown("---")
st.markdown("Lulu - Math Wizard")
