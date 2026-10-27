# Testing & Quality Assurance Plan

## 1. Overview
This document details the functional test suite, verification checklist, and error handling test cases designed for validating the **Plant Disease Detection AI** Streamlit application (`app/app.py`).

---

## 2. Functional Test Cases & Validation Checklist

| Test ID | Test Scenario | Description / Input | Expected Result | Status |
|---------|---------------|---------------------|-----------------|--------|
| **TC-01** | Model File Loading | App startup with valid `models/best_plant_model.keras` | Model loads successfully via `@st.cache_resource`; `✅ Model Loaded` indicator in sidebar. | Verified |
| **TC-02** | Missing Model Handling | App startup when model file path is missing/invalid | Catches `FileNotFoundError`, displays red error box with path guidance, halts execution cleanly. | Verified |
| **TC-03** | Step 1 Location & Weather | Input `Bhopal, India`, `Greater Noida`, `Delhi`, `Mumbai` | Resolves location name, country, lat/lon via Open-Meteo Geocoding API and displays compact weather metrics cards. | Verified |
| **TC-04** | Step 1 Invalid Location | Input invalid location `xyz123invalidcityname99` | Displays warning box: *"⚠️ Location not found. Please enter a valid city or postal code."* without app crash. | Verified |
| **TC-05** | Step 1 Missing Location | Leave location input empty | Prompts warning: *"Please enter your location first."* | Verified |
| **TC-06** | Weather API Failure Handling | Weather service timeout or connection error | Displays warning banner while allowing user to proceed to leaf image upload & disease prediction. | Verified |
| **TC-07** | Step 2 Leaf Image Upload | Upload clear `.jpg`, `.jpeg`, `.png` leaf image | Previews image, displays supported crops (Pepper, Potato, Tomato), enables Step 3 AI diagnosis. | Verified |
| **TC-08** | Step 2 Invalid / Corrupted Image | Upload unreadable file | Catches PIL exception, displays red error message `❌ Failed to load uploaded image`, halts gracefully. | Verified |
| **TC-09** | Step 3 Image Preprocessing | RGB conversion, 224x224 resize, float32 conversion | Generates NumPy tensor of shape `(1, 224, 224, 3)` with values suitable for MobileNetV3 input. | Verified |
| **TC-10** | Step 3 Primary Diagnosis | Model output evaluation | Displays user-friendly disease title, high-contrast confidence percentage, and correct card style (healthy vs. diseased). | Verified |
| **TC-11** | Step 3 Confidence Badges | Evaluate confidence percentage threshold | Renders `🟢 High Confidence` (≥80%), `🟡 Moderate Confidence` (60–79.9%), or `🔴 Low Confidence` (<60%) badge correctly. | Verified |
| **TC-12** | Step 3 Top-3 Rankings & Chart | Extract top 3 probabilities | Displays rank medals (🥇, 🥈, 🥉), progress bars, and a clean horizontal bar chart. | Verified |
| **TC-13** | Step 3 15-Class Breakdown | Open expandable section | Lists all 15 trained classes in exact order with their corresponding probability percentages. | Verified |
| **TC-14** | Step 4 Disease + Weather Analysis | Synthesize predicted disease + current weather | Explains relationship between detected disease and weather factors using educational wording. | Verified |
| **TC-15** | Step 4 Monitoring Level Card | Evaluate humidity, rain prob, and rain volume | Renders `🟢 Lower Monitoring`, `🟡 Moderate Monitoring`, or `🔴 Higher Monitoring` badge and explanation. | Verified |
| **TC-16** | Step 5 Preventive Measures | Evaluate predicted Top-1 class | Displays disease-specific cultural, sanitation, and hygiene preventive checklist. | Verified |
| **TC-17** | Plant Health Summary Card | Render bottom summary container | Displays Location, Detected Condition, AI Confidence, Weather Status, Monitoring Level, and Final Rec. | Verified |
| **TC-18** | Educational Disclaimers | Render disclaimer card | Clarifies AI diagnosis is preliminary and weather analysis provides general environmental monitoring guidance. | Verified |
| **TC-19** | Reset / Analyze Another Image | Click `🔄 Analyze Another Image` | Resets leaf upload state while preserving location data in session state. | Verified |

---

## 3. Tests Performed vs Test Checklist

### 3.1 Tests Performed (Empirically Executed)
- **Python Syntax Compilation:** Compiled `app/app.py` via `py -3.11 -m py_compile app/app.py` (0 syntax errors).
- **Automated Guided Workflow Integration Script:** Executed `test_weather_integration.py` against Python 3.11 environment:
  - Verified 15 `CLASS_NAMES` count and order.
  - Verified `models/best_plant_model.keras` existence and timestamp safety.
  - Verified Step 1 location geocoding and weather retrieval for `Bhopal`.
  - Verified crop detection helper for Pepper, Potato, and Tomato.
  - Verified Step 4 Disease + Weather synthesis text and risk monitoring levels (`Higher Monitoring` and `Lower Monitoring`).
  - Verified Step 5 disease-specific preventive measures generation for fungal and healthy plant conditions.

### 3.2 Verification Checklist & Test Cases
- **Interactive UI Verification:** Launch Streamlit application using `py -3.11 -m streamlit run app/app.py` to manually verify browser layout rendering, workflow progress bar, metric cards, and leaf upload workflows.
