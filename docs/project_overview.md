# Project Overview: Professional GUI & Plant Disease Detection System

## 1. Executive Summary
The **Plant Disease Detection AI** system is a polished computer-vision and agricultural weather analysis application structured into a guided 5-step user workflow with professional dark-mode CSS styling. Built as a B.Tech Capstone Project, the system leverages a pre-trained **MobileNetV3Large** deep learning model, an **Open-Meteo Weather API integration**, a **Disease-Weather Synthesis Engine**, and a **Streamlit** web interface to deliver rapid, reliable, and user-friendly agricultural decision support.

The application targets three key agricultural crops—**Pepper (Bell)**, **Potato**, and **Tomato**—classifying leaf samples into **15 distinct disease and healthy categories** while evaluating microclimatic weather risks and providing disease-specific preventive care recommendations.

---

## 2. Problem Statement
Plant diseases and adverse microclimatic conditions (elevated relative humidity, excessive precipitation, leaf wetness, temperature stress) are primary causes of crop yield loss, economic instability for farmers, and food insecurity worldwide. Traditional disease identification relies on visual inspection by agricultural experts, which is often:
- **Time-Consuming:** Delayed diagnosis allows pathogens to spread rapidly.
- **Geographically Limited:** Domain experts are unavailable in many remote rural regions.
- **Disconnected from Environmental Factors:** Diagnostic tools frequently ignore ambient weather patterns that govern pathogen multiplication.

An integrated system combining automated deep learning diagnosis with real-time location weather risk analysis provides an accessible, proactive decision-support tool.

---

## 3. Professional GUI & Step-by-Step Workflow Architecture

1. **Compact Top Workflow Bar:** Visual progress indicator (`📍 1. Location` → `📷 2. Leaf Image` → `🔬 3. AI Diagnosis` → `🌦️ 4. Weather Analysis` → `🌱 5. Prevention`).
2. **📍 STEP 1 — PLANT LOCATION:** Location input with `Bhopal, India` placeholder, `🌦️ Check Weather` button, and compact metric cards (Location, Temperature, Humidity, Rain Probability, Wind Speed).
3. **📷 STEP 2 — UPLOAD LEAF IMAGE:** Clean uploader with leaf upload tips and crop support badges (`🌶️ Pepper`, `🥔 Potato`, `🍅 Tomato`).
4. **🔬 STEP 3 — AI DIAGNOSIS:** MobileNetV3 model inference displaying Top-1 predicted condition, confidence level badge (`🟢 High Confidence`, `🟡 Moderate Confidence`, `🔴 Low Confidence`), Top-3 rankings, and probability distribution charts.
5. **🌦️ STEP 4 — DISEASE + WEATHER ANALYSIS:** Disease-specific contextual synthesis explaining the relationship between the predicted condition and live weather parameters alongside a Weather-Based Monitoring Level card (`🟢 Lower Monitoring`, `🟡 Moderate Monitoring`, `🔴 Higher Monitoring`).
6. **🌱 STEP 5 — PREVENTIVE MEASURES:** Tailored, disease-specific cultural, sanitation, and hygiene recommendations.
7. **📋 PLANT HEALTH SUMMARY:** Consolidated summary card at the bottom displaying Location, Detected Condition, AI Confidence, Weather Status, Monitoring Level, and Recommended Action.

---

## 4. Target Crops & Scope
The application is specifically tailored for:
- 🫑 **Pepper (Bell):** Bacterial Spot, Healthy
- 🥔 **Potato:** Early Blight, Late Blight, Healthy
- 🍅 **Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-Spotted Spider Mite, Target Spot, Tomato Mosaic Virus, Yellow Leaf Curl Virus, Healthy

---

## 5. Technology Stack Summary
- **Language:** Python 3.11.9
- **Deep Learning Framework:** TensorFlow 2.21.0 / Keras (`.keras` format)
- **Frontend Framework:** Streamlit 1.51.1
- **Weather API Services:** Open-Meteo Geocoding & Forecast APIs (Python `requests`)
- **Data & Image Libraries:** NumPy 2.4.6, Pandas, Pillow (PIL)
- **Model File:** `models/best_plant_model.keras`
