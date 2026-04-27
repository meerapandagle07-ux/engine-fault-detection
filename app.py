import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# ---------------- UI DESIGN ---------------- #

st.set_page_config(page_title="Engine Fault Detection", layout="centered")

# Image / Logo
st.image("https://cdn-icons-png.flaticon.com/512/743/743922.png", width=120)

# Title
st.title("🚗 AI-Based Engine Health Monitoring System")

st.write("This system predicts engine condition using Temperature (°C), Vibration (mm/s), and Sound (dB).")

st.markdown("---")

# ---------------- LOAD DATA ---------------- #

data = pd.read_excel("engine_fault_dataset_10000.xlsx")

X = data[['Temperature', 'Vibration', 'Sound']]
y = data['Condition']

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# ---------------- INPUT SECTION ---------------- #

st.subheader("🔧 Input Parameters")

temperature = st.slider("🌡 Temperature (°C)", 60, 120)
vibration = st.slider("🔧 Vibration (mm/s)", 0, 100)
sound = st.slider("🔊 Sound (dB)", 0, 100)

st.markdown("---")

# ---------------- THRESHOLD WARNINGS ---------------- #

st.subheader("⚠️ Real-Time Monitoring")

if temperature > 100:
    st.error("🔥 High Temperature! Risk of overheating")
elif temperature > 85:
    st.warning("⚠️ Temperature slightly high")

if vibration > 30:
    st.error("🔧 High Vibration! Possible mechanical issue")
elif vibration > 15:
    st.warning("⚠️ Moderate vibration detected")

if sound > 80:
    st.error("🔊 High Noise! Possible engine fault")
elif sound > 60:
    st.warning("⚠️ Noise level increasing")

st.markdown("---")

# ---------------- PREDICTION ---------------- #

if st.button("🔍 Predict Engine Condition"):
    prediction = model.predict([[temperature, vibration, sound]])
    probability = model.predict_proba([[temperature, vibration, sound]])

    confidence = max(probability[0]) * 100

    st.subheader("🤖 Prediction Result")

    if prediction[0] == 0:
        st.success(f"✅ Normal Engine ({confidence:.2f}% confident)")
    elif prediction[0] == 1:
        st.warning(f"⚠️ Maintenance Required ({confidence:.2f}% confident)")
    else:
        st.error(f"🚨 Fault / Misfire Detected ({confidence:.2f}% confident)")

    # ---------------- GAUGE CHART ---------------- #

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Prediction Confidence (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
        }
    ))

    st.plotly_chart(fig)

# ---------------- FOOTER ---------------- #

st.markdown("---")
st.write("👩‍💻 Developed by Meera Pandagale | AIML Project")
