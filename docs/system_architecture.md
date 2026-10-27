# System Architecture & Component Specifications

## 1. Architectural Overview

The **Plant Disease Detection AI** system follows a guided 5-step sequential architecture with professional CSS card styling, powered by **Streamlit**, **TensorFlow Keras**, and **Open-Meteo REST APIs**. The application enforces explicit user workflow dependencies while combining computer-vision model inference with live location weather risk synthesis.

```
+-------------------------------------------------------------------+
|                         USER INTERFACE                            |
|                 Streamlit Web Dashboard (app/app.py)              |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|                   TOP WORKFLOW PROGRESS BAR                       |
|  📍 1. Location → 📷 2. Leaf Image → 🔬 3. AI Diagnosis           |
|  → 🌦️ 4. Weather Analysis → 🌱 5. Prevention                      |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|                 STEP 1: STEP 1 — PLANT LOCATION                   |
|  Location Input (Bhopal, India) ──► Open-Meteo Geocoding & Forecast|
|  ──► Compact Metric Cards (Temp, Humidity, Rain Prob, Wind Speed) |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|               STEP 2: STEP 2 — UPLOAD LEAF IMAGE                  |
|  Leaf File Upload (.jpg, .jpeg, .png) ──► Supported Crops Check   |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|                 STEP 3: STEP 3 — AI DIAGNOSIS                     |
|  PIL Load ──► RGB 224x224 ──► MobileNetV3 Keras Inference        |
|  ──► Diagnosis Card, High/Mod/Low Badges & Top-3 Charts          |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|         STEP 4: STEP 4 — DISEASE + WEATHER ANALYSIS               |
|  Synthesize (Predicted Disease + Weather Parameters)              |
|  ──► Weather-Based Monitoring Level (Lower / Moderate / Higher)   |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|           STEP 5: STEP 5 — PREVENTIVE MEASURES                    |
|  Disease-Specific Cultural & Sanitation Checklist                 |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|                    PLANT HEALTH SUMMARY CARD                      |
|  Consolidated Location, Condition, Confidence, Weather & Action   |
+-------------------------------------------------------------------+
```

---

## 2. Component Specifications

### 2.1 Web Dashboard Layer (`app/app.py`)
- **Framework:** Streamlit 1.51.1
- **Styling:** Custom dark-mode CSS with responsive cards, rounded borders (`border-radius: 8px-10px`), status badges, and compact metrics.
- **Session State Management:** Stores `st.session_state["user_location"]`, `st.session_state["geocoded_info"]`, and `st.session_state["weather_info"]` to persist weather inputs across leaf image re-uploads.

### 2.2 Model Caching & Loading Module
- **Function:** `load_trained_model()`
- **Decorator:** `@st.cache_resource` (ensures single memory load across user sessions).
- **File Location:** `models/best_plant_model.keras`

### 2.3 Image Preprocessing Pipeline
Input image transformations follow these steps:
1. **Format Check:** Accepts `.jpg`, `.jpeg`, and `.png` uploaded files via `st.file_uploader`.
2. **RGB Conversion:** `raw_image.convert("RGB")` ensures consistent 3-channel tensor input.
3. **Spatial Resizing:** `rgb_image.resize((224, 224))` matches MobileNetV3 input resolution.
4. **Tensor Normalization:** `np.array(resized_image, dtype=np.float32)` converts pixels into floating-point values.
5. **Batch Expansion:** `np.expand_dims(img_array, axis=0)` shapes tensor to `(1, 224, 224, 3)`.

### 2.4 Open-Meteo Weather Subsystem
- **Geocoding API:** `geocode_location(query)`
  - URL: `https://geocoding-api.open-meteo.com/v1/search?name={query}&count=1&language=en&format=json`
  - Function: Resolves location query string into location name, country, latitude, and longitude.
- **Forecast API:** `get_weather_forecast(latitude, longitude)`
  - URL: `https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=...&daily=...`
  - Function: Retrieves current metrics (`temperature_2m`, `relative_humidity_2m`, `precipitation`, `rain`, `precipitation_probability`, `wind_speed_10m`).
- **Caching:** Both functions use `@st.cache_data(ttl=1800)` to optimize network performance.

### 2.5 Disease + Weather Synthesis Analysis & Risk Engine
- **Synthesis Engine:** `analyze_disease_and_weather(crop_name, top_1_raw, top_1_friendly, current_weather)`
  - Combines predicted crop, predicted disease name, confidence level, and live weather variables.
  - Explains how microclimatic factors influence pathogen germination and spread.
  - Assigns monitoring concern levels (`Lower Monitoring`, `Moderate Monitoring`, `Higher Monitoring`).
- **Preventive Generator:** `get_disease_preventive_measures(top_1_raw)`
  - Maps 15 disease classes to contextual cultural and sanitation measures checklists.

---

## 3. Output Specification & Class Mapping

The output tensor produced by `model.predict(img_batch, verbose=0)` is a 15-element Softmax probability vector mapped to the exact class order:

```python
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_healthy"
]
```
