import streamlit as st
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from email.mime.text import MIMEText
import smtplib
from datetime import datetime
import os

st.set_page_config(page_title="BTC Predictor", layout="centered")

# ========== CONFIGURACIÓN EMAIL ==========
EMAIL_SENDER = 'nahuelgonzaleez18@gmail.com'
EMAIL_PASSWORD = 'Hh54915550!'  # Contraseña de aplicación
EMAIL_RECEIVER = 'lucianogonzalez3944@gmail.com'
# ==========================================

# 1. Cargar datos
archivo = "bitcoin_2009-06-01_2025-06-13.csv"
df = pd.read_csv(archivo, parse_dates=["Date"], index_col="Date")
df = df.rename(columns=lambda x: x.strip().capitalize())

# 2. Indicadores técnicos
df['RSI'] = ta.rsi(df['Close'], length=14)
macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
df = df.join(macd)

# 3. Target de predicción
df['Tomorrow'] = df['Close'].shift(-1)
df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)

df = df.dropna()

features = ['RSI', 'MACD_12_26_9', 'MACDs_12_26_9']
X = df[features]
y = df['Target']

# 4. Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Evaluación
st.subheader("📊 Evaluación del modelo")
st.text(classification_report(y_test, y_pred))
st.write(f"**Precisión:** {accuracy_score(y_test, y_pred):.2%}")

# 6. Predicción actual
latest = df[features].iloc[-1:]
prediction = model.predict(latest)[0]
mensaje = "🔮 Predicción para mañana: **SUBIRÁ 📈**" if prediction == 1 else "🔮 Predicción para mañana: **BAJARÁ/MANTIENE 📉**"
st.subheader(mensaje)

# 7. Mostrar gráfico
st.subheader("📈 Gráfico del precio de BTC")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['Close'], label='Precio de Cierre', color='black')
ax.set_title('Precio BTC con Indicadores')
ax.grid(True)
st.pyplot(fig)

# 8. Guardar predicción
os.makedirs("registros", exist_ok=True)
registro_path = "registros/predicciones_btc.csv"
hoy = datetime.now().strftime('%Y-%m-%d')
precio_actual = df['Close'].iloc[-1]

nuevo_registro = pd.DataFrame([{
    "Fecha": hoy,
    "Precio Actual": precio_actual,
    "RSI": latest['RSI'].values[0],
    "MACD": latest['MACD_12_26_9'].values[0],
    "Señal": latest['MACDs_12_26_9'].values[0],
    "Predicción": "SUBE" if prediction == 1 else "BAJA/MANTIENE"
}])

if os.path.exists(registro_path):
    nuevo_registro.to_csv(registro_path, mode='a', header=False, index=False)
else:
    nuevo_registro.to_csv(registro_path, index=False)

st.success("✅ Predicción guardada en el historial")

# 9. Botón para enviar alerta por email
if st.button("✉️ Enviar alerta por email"):
    try:
        msg = MIMEText(mensaje)
        msg['Subject'] = '🔔 Alerta de Predicción BTC'
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)

        st.success("✅ Alerta enviada por email.")
    except Exception as e:
        st.error(f"❌ Error al enviar el correo: {e}")

# 10. Mostrar historial
if os.path.exists(registro_path):
    st.subheader("📅 Historial de predicciones")
    historial = pd.read_csv(registro_path)
    st.dataframe(historial.tail(10), use_container_width=True)

