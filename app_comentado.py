# Importa a biblioteca Streamlit para criar a interface web
import streamlit as st
# Importa o PyTorch (framework de redes neurais)
import torch
# Importa o módulo de redes neurais do PyTorch
import torch.nn as nn
# Importa o NumPy para operações matemáticas com arrays
import numpy as np
# Importa o Pandas para manipulação e leitura da planilha
import pandas as pd
# Importa o StandardScaler para normalização dos dados
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------------------------------------

# Define configurações da aba do navegador
st.set_page_config(
    page_title="Preditor de Sinal",   # Nome da aba
    layout="centered",                # Layout centralizado
    page_icon="📡"                    # Ícone da aba
)

# Exibe o título principal do aplicativo
st.title("📡 Preditor de Sinal")

# Exibe uma descrição abaixo do título
st.write("Informe os parâmetros abaixo para prever o valor do campo elétrico (dBm).")


# -------------------------------------------------
# FUNÇÃO PARA CARREGAR E PREPARAR OS DADOS
# -------------------------------------------------

# Faz cache dos dados para evitar baixar a planilha toda vez
@st.cache_data
def carregar_dados():

    # Link da planilha pública em formato CSV
    url = "https://docs.google.com/spreadsheets/d/1HcvCK4XDx3I5U6wkq7ea0v4POH5o-jtD2ZoTB-xbj6E/export?format=csv"
    
    # Lê os dados da planilha usando Pandas
    df = pd.read_csv(url)

    # Remove espaços extras dos nomes das colunas
    df.columns = df.columns.str.strip()

    # Define as variáveis de entrada (features)
    # Aqui usamos as 3 variáveis separadamente
    X = df[['Distância (cm)', 'Altura (cm)', 'Potência (mW)']].values

    # Define a variável alvo (target)
    y = df[['Campo (dbm)']].values

    # Cria objeto para normalizar entradas
    scaler_x = StandardScaler()

    # Cria objeto para normalizar saída
    scaler_y = StandardScaler()

    # Normaliza as variáveis de entrada
    X_norm = scaler_x.fit_transform(X)

    # Normaliza a variável de saída
    y_norm = scaler_y.fit_transform(y)

    # Retorna dados normalizados e os normalizadores
    return X_norm, y_norm, scaler_x, scaler_y


# -------------------------------------------------
# FUNÇÃO PARA TREINAR O MODELO
# -------------------------------------------------

# Faz cache do modelo para que ele treine apenas uma vez
@st.cache_resource
def treinar_modelo():

    # Carrega os dados já normalizados
    X_norm, y_norm, scaler_x, scaler_y = carregar_dados()

    # Converte as entradas para tensor do PyTorch
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    # Converte a saída para tensor do PyTorch
    y_tensor = torch.tensor(y_norm, dtype=torch.float32)

    # Define a arquitetura da rede neural
    model = nn.Sequential(

        # Primeira camada: 3 entradas → 16 neurônios
        nn.Linear(3, 16),

        # Função de ativação ReLU (introduz não-linearidade)
        nn.ReLU(),

        # Segunda camada: 16 neurônios → 8 neurônios
        nn.Linear(16, 8),

        # Outra função ReLU
        nn.ReLU(),

        # Camada final: 8 neurônios → 1 saída
        nn.Linear(8, 1)
    )

    # Define a função de perda (Erro Quadrático Médio)
    criterion = nn.MSELoss()

    # Define o otimizador Adam (mais eficiente que SGD)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Loop de treinamento (1500 épocas)
    for _ in range(1500):

        # Faz a predição com os dados de treino
        pred = model(X_tensor)

        # Calcula o erro entre predição e valor real
        loss = criterion(pred, y_tensor)

        # Calcula os gradientes
        loss.backward()

        # Atualiza os pesos da rede
        optimizer.step()

        # Zera os gradientes acumulados
        optimizer.zero_grad()

    # Retorna o modelo treinado e os normalizadores
    return model, scaler_x, scaler_y


# -------------------------------------------------
# FUNÇÃO PARA FAZER PREVISÃO
# -------------------------------------------------

def prever(distancia, altura, potencia, model, scaler_x, scaler_y):

    # Cria array com os valores informados pelo usuário
    entrada = np.array([[distancia, altura, potencia]])

    # Normaliza a nova entrada usando o mesmo scaler do treino
    entrada_norm = scaler_x.transform(entrada)

    # Desativa cálculo de gradiente (modo inferência)
    with torch.no_grad():

        # Converte para tensor e faz a predição
        pred_norm = model(torch.tensor(entrada_norm, dtype=torch.float32)).numpy()

    # Converte o valor previsto de volta para escala real (dBm)
    return scaler_y.inverse_transform(pred_norm)[0][0]


# -------------------------------------------------
# INTERFACE DO USUÁRIO
# -------------------------------------------------

# Cria três colunas para organizar os campos de entrada
col1, col2, col3 = st.columns(3)

# Campo para entrada da distância
with col1:
    distancia = st.number_input("Distância (cm)", min_value=0.0)

# Campo para entrada da altura
with col2:
    altura = st.number_input("Altura (cm)", min_value=0.0)

# Campo para selecionar potência
with col3:
    potencia = st.selectbox("Potência (mW)", [100, 300])

# Carrega o modelo treinado
model, scaler_x, scaler_y = treinar_modelo()

# Botão para realizar a previsão
if st.button("🔍 Prever Sinal"):

    # Chama a função de previsão
    resultado = prever(distancia, altura, potencia, model, scaler_x, scaler_y)

    # Mostra o resultado
    st.success(f"📡 Sinal predito: **{resultado:.2f} dBm**")

# -------------------------------------------------
# RODAPÉ
# -------------------------------------------------

st.markdown("---")
col_logo, col_texto = st.columns([1, 3])

# Mostra a logo do IFPB
with col_logo:
    st.image("logo_ifpb.png", width=120)

# Mostra o texto institucional ao lado da logo
with col_texto:
    st.markdown(
        "<div style='font-size: 14px; color: gray; margin-top: 35px;'>"
        "Projeto viabilizado pelo CNPq, PIBIC e pelo IFPB"
        "</div>",
        unsafe_allow_html=True
    )
