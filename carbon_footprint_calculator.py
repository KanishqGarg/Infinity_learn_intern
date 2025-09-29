# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64

# ---------- Predefined habits (approx. kg CO2e per day) ----------
HABIT_DB = {
    "Walk short trip (instead of car)": 0.0,
    "Bike short trip (instead of car)": 0.0,
    "Car (short daily trip)": 4.6,
    "Public bus (daily)": 1.5,
    "Eat meat (daily)": 2.5,
    "Mostly vegetarian (daily)": 0.6,
    "Use TV 2 hours": 0.3,
    "Use computer 2 hours": 0.4,
    "Leave lights on (evening)": 0.2,
    "Use AC (daily, few hours)": 3.0,
    "Take a short shower (water+heat)": 0.9,
    "Long shower (hot)": 2.0,
    "Laundry (per day average)": 0.5,
    "Buy new clothes (amortized daily)": 0.8,
    "Use single-use plastic bottle": 0.1,
    "Plant-based lunch (eco choice)": 0.2,
    "Recycle waste (good habit)": -0.1,  # negative means reduction
}

# ---------- Helper functions ----------
def calculate_impact(selected_habits):
    daily = 0.0
    breakdown = {}
    for h in selected_habits:
        val = HABIT_DB.get(h, 0.0)
        breakdown[h] = val
        daily += val
    monthly = daily * 30
    annual = daily * 365
    return daily, monthly, annual, breakdown

def feedback_and_tips(annual_kg):
    if annual_kg < 500:
        msg = "🌟 Amazing! Very low footprint for a year."
        tips = [
            "Keep walking and biking!",
            "Share tips with friends about saving energy."
        ]
    elif annual_kg < 2000:
        msg = "🙂 Good job! Room to make a few greener swaps."
        tips = [
            "Try meat-free meals twice a week.",
            "Turn off lights and unplug chargers when not in use."
        ]
    else:
        msg = "⚠️ Your footprint is high — small changes can help a lot!"
        tips = [
            "Replace short car trips with walking or bus.",
            "Shorten showers and reduce AC usage where possible.",
            "Eat more plant-based meals."
        ]
    return msg, tips

def df_to_csv_bytes(df):
    csv = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:text/csv;base64,{b64}" download="carbon_report.csv">Download CSV report</a>'

# ---------- Streamlit UI ----------
st.title("🌱 Kids Carbon Footprint Calculator")

st.write("Select the habits you do daily:")

# Multi-select for habits
selected_habits = st.multiselect("Habits", list(HABIT_DB.keys()))

if st.button("Calculate 🌍"):
    if not selected_habits:
        st.warning("Please select at least one habit.")
    else:
        daily, monthly, annual, breakdown = calculate_impact(selected_habits)

        st.subheader("🧾 Results")
        st.write(f"**Daily carbon impact:** {daily:.2f} kg CO₂e")
        st.write(f"**Monthly (est):** {monthly:.2f} kg CO₂e")
        st.write(f"**Annual (est):** {annual:.2f} kg CO₂e")

        msg, tips = feedback_and_tips(annual)
        st.success(msg)
        st.write("💡 Tips:")
        for t in tips:
            st.write(f"- {t}")

        # Dataframe
        rows = [{"habit": k, "daily_kgCO2e": float(v)} for k,v in breakdown.items()]
        df = pd.DataFrame(rows).sort_values("daily_kgCO2e", ascending=False)
        st.dataframe(df.style.format({"daily_kgCO2e":"{:.2f}"}))

        # Bar chart
        fig, ax = plt.subplots(figsize=(8,3))
        ax.bar(df['habit'], df['daily_kgCO2e'], color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("kg CO₂e per day")
        plt.title("Daily contribution by habit")
        plt.tight_layout()
        st.pyplot(fig)

        # CSV download
        st.markdown(df_to_csv_bytes(df), unsafe_allow_html=True)
