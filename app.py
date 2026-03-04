import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------
st.set_page_config(
    page_title="Preditor de Sinal",
    layout="centered",
    page_icon="📡"
)

st.title("📡 Preditor de Sinal")
st.write("Informe os parâmetros abaixo para prever o valor do campo elétrico (dBm).")

# ---------------------------------
# CARREGAR DADOS
# ---------------------------------
@st.cache_data
def carregar_dados():
    url = "https://docs.google.com/spreadsheets/d/1HcvCK4XDx3I5U6wkq7ea0v4POH5o-jtD2ZoTB-xbj6E/export?format=csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()

    X = df[['Distância (cm)', 'Altura (cm)', 'Potência (mW)']].values
    y = df[['Campo (dbm)']].values

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_norm = scaler_x.fit_transform(X)
    y_norm = scaler_y.fit_transform(y)

    return X_norm, y_norm, scaler_x, scaler_y

# ---------------------------------
# TREINAR MODELO
# ---------------------------------
@st.cache_resource
def treinar_modelo():
    X_norm, y_norm, scaler_x, scaler_y = carregar_dados()

    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(3, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(1500):
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return model, scaler_x, scaler_y

# ---------------------------------
# FUNÇÃO DE PREVISÃO
# ---------------------------------
def prever(distancia, altura, potencia, model, scaler_x, scaler_y):
    entrada = np.array([[distancia, altura, potencia]])
    entrada_norm = scaler_x.transform(entrada)

    with torch.no_grad():
        pred_norm = model(torch.tensor(entrada_norm, dtype=torch.float32)).numpy()

    return scaler_y.inverse_transform(pred_norm)[0][0]

# ---------------------------------
# INTERFACE
# ---------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    distancia = st.number_input("Distância (cm)", min_value=0.0)

with col2:
    altura = st.number_input("Altura (cm)", min_value=0.0)

with col3:
    potencia = st.selectbox("Potência (mW)", [100, 300])

model, scaler_x, scaler_y = treinar_modelo()

if st.button("🔍 Prever Sinal"):
    resultado = prever(distancia, altura, potencia, model, scaler_x, scaler_y)

    st.success(f"📡 Sinal predito: **{resultado:.2f} dBm**")

# ---------------------------------
# RODAPÉ INSTITUCIONAL
# ---------------------------------
st.markdown("---")

col_logo, col_texto = st.columns([1, 3])

with col_logo:
    st.image("logo_ifpb.png", width=120)

with col_texto:
    st.markdown(
        "<div style='font-size: 14px; color: gray; margin-top: 35px;'>"
        "Projeto viabilizado pelo CNPq, PIBIC e pelo IFPB"
        "</div>",
        unsafe_allow_html=True
    )
