import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Smart Energy Match", layout="centered")

st.title("⚡ Smart Energy Match – AI Прогноза за Продажба на Ел. Енергия")

st.markdown("""
Това приложение използва симулирани борсови цени и AI алгоритъм, за да ти предложи **оптимален момент за продажба** на електроенергия, произведена от фотоволтаици.
""")

# --- Въвеждане на данни от потребителя ---
st.subheader("Въведи данни за производство")
produced_kwh = st.number_input("Произведена ел. енергия (kWh)", min_value=0.0, step=0.1)
battery_capacity = st.number_input("Свободен капацитет на батерията (kWh)", min_value=0.0, step=0.1)
start_time = st.time_input("Начален час на прогнозата", value=datetime.time(8, 0))

# --- Генериране на примерни борсови цени ---
hours = pd.date_range(datetime.datetime.combine(datetime.date.today(), start_time), periods=12, freq='H')
base_price = 150 + np.random.randn(12) * 10  # средна цена ~150 лв/MWh с шум

# --- AI модел: проста линейна регресия за тренд ---
X = np.array(range(len(hours))).reshape(-1, 1)
y = base_price
model = LinearRegression().fit(X, y)
predicted_prices = model.predict(X)

# --- Създаване на датафрейм ---
data = pd.DataFrame({
    "Час": hours.strftime('%H:%M'),
    "Борсова цена (лв/MWh)": base_price.round(2),
    "Прогнозна цена (AI)": predicted_prices.round(2)
})
data["Препоръка"] = np.where(predicted_prices > base_price, "Задръж", "Продай")

st.subheader("📊 Прогнозен анализ")
st.dataframe(data, use_container_width=True)

# --- Графика ---
st.line_chart(data.set_index("Час")[["Борсова цена (лв/MWh)", "Прогнозна цена (AI)"]])

# --- Общ съвет ---
st.subheader("🤖 AI Съвет")
recommendation = data["Препоръка"].value_counts().idxmax()
if recommendation == "Задръж":
    st.success("AI препоръка: Изчакай – цените ще се покачат през следващите часове.")
else:
    st.warning("AI препоръка: Продай сега – не се очаква повишение на цените.")
