# Plant Disease Detection AI

An AI-powered web application for automated plant disease classification using deep transfer learning (**MobileNetV3Large**), real-time **Open-Meteo Weather Analysis**, and **Streamlit**. Designed with a professional GUI/UX and a guided step-by-step workflow for **Pepper (Bell)**, **Potato**, and **Tomato** crops.

---

## 📌 Overview

Plant diseases and microclimatic weather conditions represent major threats to agricultural production. The **Plant Disease Detection AI** system provides an interactive, guided 5-step workflow that integrates local weather analysis, deep learning leaf photo diagnosis, environmental disease-weather synthesis, monitoring risk levels, and disease-specific preventive guidance.

---

## 🚀 Step-by-Step User Workflow

1. **📍 STEP 1 — PLANT LOCATION:** Enter the location where the plant is growing (e.g. Bhopal, India) and click `🌦️ Check Weather` to fetch live metrics (Temperature, Humidity, Rain Probability, Wind Speed).
2. **📷 STEP 2 — UPLOAD LEAF IMAGE:** Upload a clear leaf photograph (`.jpg`, `.jpeg`, `.png`) for supported crops: Pepper (Bell), Potato, and Tomato.
3. **🔬 STEP 3 — AI DIAGNOSIS:** Runs pre-trained MobileNetV3Large model inference, displaying predicted condition, high-contrast confidence level badge (`🟢 High Confidence`, `🟡 Moderate Confidence`, `🔴 Low Confidence`), Top-3 medal rankings (🥇, 🥈, 🥉), and interactive probability distribution charts.
4. **🌦️ STEP 4 — DISEASE + WEATHER ANALYSIS:** Synthesizes the predicted crop condition with live weather variables to explain environmental disease pressure and display a prominent Weather-Based Monitoring Level (`🟢 Lower Monitoring`, `🟡 Moderate Monitoring`, `🔴 Higher Monitoring`).
5. **🌱 STEP 5 — PREVENTIVE MEASURES:** Displays tailored, disease-specific cultural, sanitation, and hygiene recommendations checklist.
6. **📋 PLANT HEALTH SUMMARY:** Displays a consolidated summary card at the bottom aggregating Location, Detected Condition, AI Confidence, Weather Status, Monitoring Level, and Recommended Action.

---

## 📋 Supported Disease Classes

The deep learning model evaluates input images against exactly **15 classes** in the following trained order:

| # | Raw Dataset Class Name | User-Friendly Display Name |
|---|------------------------|---------------------------|
| 1 | `Pepper__bell___Bacterial_spot` | Pepper (Bell) - Bacterial Spot |
| 2 | `Pepper__bell___healthy` | Pepper (Bell) - Healthy |
| 3 | `Potato___Early_blight` | Potato - Early Blight |
| 4 | `Potato___Late_blight` | Potato - Late Blight |
| 5 | `Potato___healthy` | Potato - Healthy |
| 6 | `Tomato_Bacterial_spot` | Tomato - Bacterial Spot |
| 7 | `Tomato_Early_blight` | Tomato - Early Blight |
| 8 | `Tomato_Late_blight` | Tomato - Late Blight |
| 9 | `Tomato_Leaf_Mold` | Tomato - Leaf Mold |
| 10 | `Tomato_Septoria_leaf_spot` | Tomato - Septoria Leaf Spot |
| 11 | `Tomato_Spider_mites_Two_spotted_spider_mite` | Tomato - Two-Spotted Spider Mite |
| 12 | `Tomato__Target_Spot` | Tomato - Target Spot |
| 13 | `Tomato__Tomato_mosaic_virus` | Tomato - Tomato Mosaic Virus |
| 14 | `Tomato__Tomato_YellowLeaf__Curl_Virus` | Tomato - Yellow Leaf Curl Virus |
| 15 | `Tomato_healthy` | Tomato - Healthy |

---

## 🛠️ Technology Stack

- **Language:** Python 3.11.9
- **Deep Learning Framework:** TensorFlow 2.21.0 / Keras (`.keras` format)
- **Web Application Framework:** Streamlit 1.51.1
- **Weather Services:** Open-Meteo Geocoding & Forecast REST APIs (Python `requests` client)
- **Numerical & Data Processing:** NumPy 2.4.6, Pandas
- **Image Handling:** Pillow (PIL)

---

## 🧠 Model Architecture

- **Base Backbone:** MobileNetV3Large (Pre-trained on ImageNet)
- **Approach:** Transfer Learning
- **Input Dimensions:** 224 × 224 × 3 RGB
- **Output:** 15 Class Softmax Probabilities
- **Model File Path:** `models/best_plant_model.keras`

---

## 💻 Installation & Running

### Step 1: Open Project Directory
```bash
cd PlantDiseaseDetection
```

### Step 2: Install Dependencies
```bash
py -3.11 -m pip install -r requirements.txt
```

### Step 3: Launch Streamlit Application
```bash
py -3.11 -m streamlit run app/app.py
```

Access the application in your web browser at `http://localhost:8501`.

---

## ⚠️ Disclaimers

> ⚠️ **Educational & Decision Support Disclaimer:**
> - AI predictions are preliminary and intended for educational/informational purposes. Weather-based insights are general environmental guidance and should not replace professional agricultural diagnosis.
