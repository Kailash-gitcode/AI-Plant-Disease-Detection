# Methodology & Professional GUI Design

## 1. Transfer Learning Approach
Training deep convolutional neural networks from scratch requires massive datasets and substantial computational resources. **Transfer Learning** overcomes these challenges by utilizing a network pre-trained on a large-scale dataset (ImageNet) and fine-tuning its features for a specific domain.

For this project, **MobileNetV3Large** was selected as the base feature extractor due to its high efficiency, low parameter count, depthwise separable convolutions, and optimal balance between classification accuracy and computational latency.

---

## 2. Dataset & Scope
The model was trained on a curated subset of plant leaf images spanning **15 disease and healthy classes** across 3 major agricultural crops:

- **Pepper (Bell):** Bacterial Spot, Healthy
- **Potato:** Early Blight, Late Blight, Healthy
- **Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-Spotted Spider Mite, Target Spot, Tomato Mosaic Virus, Yellow Leaf Curl Virus, Healthy

---

## 3. GUI Layout & 5-Step Pipeline Methodology

The application structures user interaction into a clean 5-step sequential layout:

```
[ Top Workflow Indicator Bar ]
  📍 1. Location → 📷 2. Leaf Image → 🔬 3. AI Diagnosis → 🌦️ 4. Weather Analysis → 🌱 5. Prevention
                                                                 │
[ STEP 1 — PLANT LOCATION ] ──► Open-Meteo API ──► Compact Metric Cards (Temp, Humidity, Rain Prob, Wind)
                                                                 │
[ STEP 2 — UPLOAD LEAF IMAGE ] ──► File Uploader & Leaf Quality Tips
                                                                 │
[ STEP 3 — AI DIAGNOSIS ] ──► MobileNetV3 Keras Inference ──► Top-1, Confidence Badge & Top-3 Charts
                                                                 │
[ STEP 4 — DISEASE + WEATHER ANALYSIS ] ──► Contextual Synthesis & Weather-Based Monitoring Level
                                                                 │
[ STEP 5 — PREVENTIVE MEASURES ] ──► Disease-Specific Cultural Checklist
                                                                 │
[ PLANT HEALTH SUMMARY ] ──► Consolidated Overview & Final Recommended Action
```

---

## 4. Model Architecture Details

```
Input Layer: (224 x 224 x 3) RGB Image
        │
        ▼
MobileNetV3Large Backbone (Pre-trained Feature Extractor)
        │
        ▼
Global Average Pooling 2D Layer
        │
        ▼
Dropout Layer (Regularization)
        │
        ▼
Dense Layer (128 units, ReLU activation)
        │
        ▼
Output Dense Layer (15 units, Softmax activation)
```

---

## 5. Location Geocoding & Weather Retrieval Methodology

1. **User Location Processing:** The user manually inputs a city or postal code (e.g. `Bhopal, India`). Automatic GPS is disabled to maintain user privacy.
2. **Open-Meteo Geocoding Lookup:** Queries `https://geocoding-api.open-meteo.com/v1/search` to extract location name, country, latitude, and longitude.
3. **Open-Meteo Weather Forecast Lookup:** Queries `https://api.open-meteo.com/v1/forecast` using extracted coordinates to retrieve current weather parameters (`temperature_2m`, `relative_humidity_2m`, `precipitation`, `rain`, `precipitation_probability`, `wind_speed_10m`).
4. **Caching & Session Persistence:** Utilizes `@st.cache_data(ttl=1800)` for API calls to prevent redundant requests and retains location data in `st.session_state`.

---

## 6. Disease + Weather Synthesis & Monitoring Level Engine

- **🔴 Higher Monitoring:**
  - `(Relative Humidity ≥ 75% AND Rain Probability ≥ 60%) OR Rain ≥ 5.0 mm OR Relative Humidity ≥ 85%`
  - *Explanation:* Higher monitoring recommended because humidity is elevated and rainfall probability is moderate to high.
- **🟡 Moderate Monitoring:**
  - `Relative Humidity 60–74% OR Rain Probability 40–59% OR Temperature > 32°C OR Temperature < 12°C`
  - *Explanation:* Moderate monitoring recommended due to ambient humidity levels or temperature conditions that may favor disease pressure.
- **🟢 Lower Monitoring:**
  - `Relative Humidity < 60% AND Rain Probability < 40% AND Mild Temperatures`
  - *Explanation:* Lower monitoring level because relative humidity and rainfall indicators are currently low.

---

## 7. Recommended Preventive Measures Methodology

Preventive guidance dynamically maps the predicted Top-1 leaf condition to specific cultural recommendations:
- **Healthy Leaves:** Focuses on preventive cultural practices (soil drainage, canopy airflow, leaf wetness monitoring).
- **Fungal Pathogens (Blights, Septoria, Leaf Mold, Target Spot):** Recommends airflow enhancement, removal of lower spotted leaves, debris disposal, and sub-canopy drip irrigation.
- **Bacterial Pathogens (Bacterial Spot):** Emphasizes avoiding handling wet foliage, removing affected material, and eliminating overhead irrigation.
- **Viral Infections (Mosaic, Leaf Curl):** Focuses on vector management (whiteflies) and tool sanitation.
- **Pests (Spider Mites):** Focuses on leaf underside inspections during hot/dry conditions.
