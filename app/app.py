from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import requests
import streamlit as st
import tensorflow as tf

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Plant Disease Detection AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional GUI Styling
st.markdown("""
    <style>
    /* Main Typography & Header */
    .main-header {
        font-size: 2.2rem;
        color: #2E7D32;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #388E3C;
        text-align: center;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .header-desc {
        font-size: 0.94rem;
        color: #424242;
        text-align: center;
        max-width: 820px;
        margin: 0 auto 1.2rem auto;
        line-height: 1.5;
    }
    
    /* Compact Top Workflow Indicator Bar */
    .workflow-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #F8F9FA;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 1.5rem;
        gap: 8px;
        flex-wrap: wrap;
    }
    .step-pill {
        font-size: 0.85rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 16px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .step-active {
        background-color: #E8F5E9;
        color: #2E7D32;
        border: 1px solid #81C784;
    }
    .step-completed {
        background-color: #C8E6C9;
        color: #1B5E20;
    }
    .step-pending {
        background-color: #EEEEEE;
        color: #757575;
    }
    .step-arrow {
        color: #BDBDBD;
        font-weight: bold;
    }

    /* Section Cards & Step Headers */
    .step-header {
        font-size: 1.25rem;
        color: #1B5E20;
        font-weight: 700;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #A5D6A7;
        padding-bottom: 6px;
    }
    
    /* Prediction Cards */
    .prediction-card-healthy {
        background-color: #E8F5E9;
        border-left: 6px solid #2E7D32;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
        color: #1B5E20;
    }
    .prediction-card-diseased {
        background-color: #FFF3E0;
        border-left: 6px solid #E65100;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
        color: #BF360C;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #424242;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #111111;
        margin-top: 4px;
        margin-bottom: 8px;
    }
    .confidence-text {
        font-size: 1.02rem;
        color: #111111;
        font-weight: 600;
    }
    
    /* Analysis & Preventive Cards */
    .analysis-card {
        background-color: #E3F2FD;
        border-left: 6px solid #1976D2;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
        color: #0D47A1;
        font-size: 0.96rem;
        line-height: 1.6;
    }
    .monitoring-card {
        background-color: #FAFAFA;
        border: 1px solid #E0E0E0;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .preventive-card {
        background-color: #F1F8E9;
        border: 1px solid #C8E6C9;
        padding: 18px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .preventive-item {
        font-size: 0.95rem;
        color: #2E7D32;
        margin-bottom: 6px;
        font-weight: 500;
    }
    .summary-card {
        background-color: #F1F8E9;
        border: 2px solid #81C784;
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .disclaimer-card {
        background-color: #FFFDE7;
        border-left: 4px solid #FBC02D;
        padding: 12px 15px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 20px;
        font-size: 0.90rem;
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS & CLASS NAMES MAPPING
# ==========================================
# Exact 15 class names in exact trained order
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

# Mapping raw class names to user-friendly formatted titles
CLASS_DISPLAY_MAP = {
    "Pepper__bell___Bacterial_spot": "Pepper (Bell) - Bacterial Spot",
    "Pepper__bell___healthy": "Pepper (Bell) - Healthy",
    "Potato___Early_blight": "Potato - Early Blight",
    "Potato___Late_blight": "Potato - Late Blight",
    "Potato___healthy": "Potato - Healthy",
    "Tomato_Bacterial_spot": "Tomato - Bacterial Spot",
    "Tomato_Early_blight": "Tomato - Early Blight",
    "Tomato_Late_blight": "Tomato - Late Blight",
    "Tomato_Leaf_Mold": "Tomato - Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Tomato - Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato - Two-Spotted Spider Mite",
    "Tomato__Target_Spot": "Tomato - Target Spot",
    "Tomato__Tomato_mosaic_virus": "Tomato - Tomato Mosaic Virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato - Yellow Leaf Curl Virus",
    "Tomato_healthy": "Tomato - Healthy"
}

# Cautious agricultural guidance dictionary
DISEASE_ADVICE = {
    "healthy": "✅ The model prediction suggests the leaf appears healthy. Continue maintaining proper watering, adequate sunlight, and soil nutrient management.",
    "Bacterial_spot": "⚠️ The model prediction suggests Bacterial Spot. Consider removing severely affected leaves, avoiding overhead watering, and consulting local agricultural guidance.",
    "Early_blight": "⚠️ The model prediction suggests Early Blight. Consider pruning lower affected leaves, improving air circulation around plants, and seeking professional agricultural advice.",
    "Late_blight": "🚨 The model prediction suggests Late Blight. It is recommended to isolate affected plants, remove diseased foliage, and consult local extension services for appropriate management.",
    "Leaf_Mold": "⚠️ The model prediction suggests Leaf Mold. Try improving air circulation, reducing ambient humidity, and consulting local agricultural guidance.",
    "Septoria_leaf_spot": "⚠️ The model prediction suggests Septoria Leaf Spot. Consider removing spotted leaves, mulching around the plant base to limit soil splash, and seeking agricultural advice.",
    "Spider_mites_Two_spotted_spider_mite": "⚠️ The model prediction suggests Two-Spotted Spider Mites. Consider washing leaf undersides or using organic neem oil sprays after consulting local agricultural experts.",
    "Target_Spot": "⚠️ The model prediction suggests Target Spot. Consider removing infected plant debris, keeping foliage dry, and consulting local agricultural advisors.",
    "Tomato_mosaic_virus": "🚨 The model prediction suggests Tomato Mosaic Virus. Viral conditions cannot be cured directly; consider isolating or removing infected plants and sanitizing garden tools.",
    "YellowLeaf__Curl_Virus": "🚨 The model prediction suggests Yellow Leaf Curl Virus. Consider managing whitefly vectors using sticky traps and consulting local agricultural specialists."
}

def format_user_friendly_name(raw_name: str) -> str:
    """Formats raw dataset class strings into clean, user-friendly labels."""
    return CLASS_DISPLAY_MAP.get(raw_name, raw_name.replace("_", " ").title())

def detect_crop_from_class(raw_name: str) -> str:
    """Extracts crop name (Pepper, Potato, Tomato) from class string."""
    name_lower = raw_name.lower()
    if "pepper" in name_lower:
        return "Pepper (Bell)"
    elif "potato" in name_lower:
        return "Potato"
    elif "tomato" in name_lower:
        return "Tomato"
    return "Plant Crop"

# ==========================================
# MODEL LOADING FUNCTION
# ==========================================
@st.cache_resource
def load_trained_model():
    """
    Loads the pre-trained TensorFlow model from models/best_plant_model.keras.
    Uses @st.cache_resource so the model is cached across Streamlit reruns.
    """
    app_dir = Path(__file__).parent.resolve()
    model_path = (app_dir / ".." / "models" / "best_plant_model.keras").resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at expected path: {model_path}")

    model = tf.keras.models.load_model(str(model_path))
    return model, str(model_path)

# ==========================================
# WEATHER MODULE (Open-Meteo API & Caching)
# ==========================================
@st.cache_data(ttl=1800, show_spinner=False)
def geocode_location(query: str):
    """
    Queries the Open-Meteo Geocoding API without requiring an API key.
    Extracts location name, country, latitude, and longitude.
    """
    if not query or not query.strip():
        return False, "Please enter your location first."

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": query.strip(),
        "count": 1,
        "language": "en",
        "format": "json"
    }
    try:
        response = requests.get(url, params=params, timeout=8)
        if response.status_code != 200:
            return False, f"Geocoding service HTTP status {response.status_code}."

        data = response.json()
        results = data.get("results")
        if not results or len(results) == 0:
            return False, "⚠️ Location not found. Please enter a valid city or postal code."

        best = results[0]
        name = best.get("name", query.strip().title())
        country = best.get("country", "")
        latitude = float(best.get("latitude"))
        longitude = float(best.get("longitude"))
        display_name = f"{name}, {country}" if country else name

        return True, {
            "display_name": display_name,
            "name": name,
            "country": country,
            "latitude": latitude,
            "longitude": longitude
        }
    except requests.exceptions.Timeout:
        return False, "⚠️ Weather location lookup timed out. Please check your network connection."
    except requests.exceptions.RequestException as e:
        return False, f"⚠️ Unable to connect to weather location service: {str(e)}"
    except Exception as e:
        return False, f"⚠️ Error processing location query: {str(e)}"

@st.cache_data(ttl=1800, show_spinner=False)
def get_weather_forecast(latitude: float, longitude: float):
    """
    Retrieves current weather conditions and 5-7 day forecast from Open-Meteo.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,precipitation_probability,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum",
        "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params, timeout=8)
        if response.status_code != 200:
            return False, f"Weather forecast service HTTP status {response.status_code}."

        data = response.json()
        current = data.get("current", {})
        daily = data.get("daily", {})

        if not current or not daily:
            return False, "⚠️ Incomplete weather data received from Open-Meteo."

        return True, {
            "current": {
                "temp": float(current.get("temperature_2m", 0.0)),
                "humidity": int(current.get("relative_humidity_2m", 0)),
                "precipitation": float(current.get("precipitation", 0.0)),
                "rain": float(current.get("rain", 0.0)),
                "rain_prob": int(current.get("precipitation_probability", 0)),
                "wind": float(current.get("wind_speed_10m", 0.0))
            },
            "daily": {
                "time": daily.get("time", []),
                "temp_max": daily.get("temperature_2m_max", []),
                "temp_min": daily.get("temperature_2m_min", []),
                "rain_prob_max": daily.get("precipitation_probability_max", []),
                "precip_sum": daily.get("precipitation_sum", [])
            }
        }
    except requests.exceptions.Timeout:
        return False, "⚠️ Weather forecast request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return False, f"⚠️ Network error retrieving weather forecast: {str(e)}"
    except Exception as e:
        return False, f"⚠️ Unexpected error retrieving weather data: {str(e)}"

# ==========================================
# DISEASE + WEATHER SYNTHESIS ENGINE
# ==========================================
def analyze_disease_and_weather(crop_name: str, top_1_raw: str, top_1_friendly: str, current_weather: dict) -> tuple[str, str, str, str]:
    """
    Step 4 Synthesis: Explains relationship between detected disease and weather.
    Returns: (analysis_text, risk_level, risk_badge_html, risk_explanation)
    """
    if not current_weather:
        analysis_text = f"Weather information is currently unavailable. The AI diagnosed **{top_1_friendly}** on your {crop_name} leaf based purely on computer vision features."
        risk_level = "Moderate Monitoring"
        risk_badge = '<span style="background-color: #FFF9C4; color: #F57F17; padding: 5px 12px; border-radius: 12px; font-weight: bold;">🟡 Moderate Monitoring</span>'
        risk_explanation = "Weather data is unavailable; maintain general crop observation."
        return analysis_text, risk_level, risk_badge, risk_explanation

    temp = current_weather.get("temp", 25.0)
    humidity = current_weather.get("humidity", 50)
    rain_prob = current_weather.get("rain_prob", 0)
    rain_mm = current_weather.get("rain", 0.0)

    is_high_moisture = (humidity >= 75 or rain_prob >= 60 or rain_mm >= 2.0)
    is_mod_moisture = (humidity >= 60 or rain_prob >= 40)
    is_hot_dry = (temp >= 30.0 and humidity <= 45)

    raw_lower = top_1_raw.lower() if top_1_raw else "healthy"

    # Risk level calculation with clear sensible thresholds
    if (humidity >= 75 and rain_prob >= 60) or (rain_mm >= 5.0) or (humidity >= 85):
        risk_level = "Higher Monitoring"
        risk_badge = '<span style="background-color: #FFCDD2; color: #B71C1C; padding: 5px 12px; border-radius: 12px; font-weight: bold;">🔴 Higher Monitoring</span>'
        risk_explanation = f"Higher monitoring recommended because humidity is elevated ({humidity}%) and rainfall probability is moderate to high ({rain_prob}%)."
    elif is_mod_moisture or temp > 32.0 or temp < 12.0:
        risk_level = "Moderate Monitoring"
        risk_badge = '<span style="background-color: #FFF9C4; color: #F57F17; padding: 5px 12px; border-radius: 12px; font-weight: bold;">🟡 Moderate Monitoring</span>'
        risk_explanation = f"Moderate monitoring recommended due to ambient humidity levels ({humidity}%) or temperature conditions that may favor disease pressure."
    else:
        risk_level = "Lower Monitoring"
        risk_badge = '<span style="background-color: #C8E6C9; color: #1B5E20; padding: 5px 12px; border-radius: 12px; font-weight: bold;">🟢 Lower Monitoring</span>'
        risk_explanation = f"Lower monitoring level because relative humidity ({humidity}%) and rainfall indicators ({rain_prob}%) are currently low."

    # Contextual analysis text
    if "healthy" in raw_lower:
        if is_high_moisture:
            analysis_text = f"The AI detected a **healthy {crop_name}** leaf. Current humidity is high ({humidity}%) and rainfall probability is moderate ({rain_prob}%). These conditions may increase the need for close monitoring of foliage because prolonged moisture can favor conditions associated with some plant diseases."
        else:
            analysis_text = f"The AI detected a **healthy {crop_name}** leaf. Current weather conditions ({temp:.1f}°C, {humidity}% humidity) are favorable for maintaining overall plant health. Continue routine watering and soil management."
    elif "early_blight" in raw_lower:
        if is_high_moisture:
            analysis_text = f"Current humidity is high ({humidity}%) and rainfall probability is moderate ({rain_prob}%). These conditions may increase the need for close monitoring of {crop_name} foliage because prolonged moisture can favor conditions associated with Early Blight. Inspect affected leaves regularly and avoid overhead watering."
        else:
            analysis_text = f"Because the AI detected **Early Blight** on your {crop_name} leaf, monitor lower foliage closely for target-like spots. Although current humidity ({humidity}%) is moderate, preventative canopy management is recommended."
    elif "late_blight" in raw_lower:
        if is_high_moisture:
            analysis_text = f"Current wet weather conditions ({rain_mm:.1f} mm rain, {humidity}% humidity) may increase the need for close monitoring of {crop_name} foliage because prolonged leaf wetness favors Late Blight progression."
        else:
            analysis_text = f"The AI detected **Late Blight** on your {crop_name} leaf. Cool or humid microclimates could favor rapid sporangia germination; maintain vigilant field monitoring."
    elif "bacterial_spot" in raw_lower:
        if is_high_moisture:
            analysis_text = f"Current warm and humid weather ({humidity}% humidity, {rain_prob}% rain probability) may enhance bacterial splash transmission across {crop_name} foliage. Avoid handling wet plants and eliminate overhead irrigation."
        else:
            analysis_text = f"The AI detected **Bacterial Spot** on your {crop_name} leaf. Monitor plants after rain or high humidity events to minimize bacterial splash spreading."
    elif "septoria" in raw_lower:
        analysis_text = f"Elevated humidity ({humidity}%) and rain likelihood ({rain_prob}%) enhance **Septoria Leaf Spot** spore splash from soil onto lower {crop_name} foliage. Remove spotted lower leaves and mulch around the base."
    elif "leaf_mold" in raw_lower:
        analysis_text = f"Tomato **Leaf Mold** is strongly favored by elevated relative humidity ({humidity}%). Ensure good canopy ventilation and avoid prolonged evening leaf wetness."
    elif "spider_mites" in raw_lower:
        if is_hot_dry:
            analysis_text = f"Hot and dry weather ({temp:.1f}°C, {humidity}% humidity) accelerates **Two-Spotted Spider Mite** reproduction and foliage stippling on {crop_name}. Inspect leaf undersides regularly."
        else:
            analysis_text = f"The AI detected **Spider Mite** damage on your {crop_name} leaf. Inspect leaf undersides regularly and maintain plant hydration to prevent drought stress."
    elif "target_spot" in raw_lower:
        analysis_text = f"Target Spot fungal development is favored by humid weather ({humidity}% humidity). Remove infected foliage showing dark concentric spots and keep foliage dry."
    elif "mosaic_virus" in raw_lower:
        analysis_text = f"The AI detected **Tomato Mosaic Virus**. Viral conditions spread via mechanical contact; current weather ({temp:.1f}°C, {humidity}% humidity) influences crop vigor."
    elif "yellowleaf__curl" in raw_lower:
        analysis_text = f"Warm weather ({temp:.1f}°C) can increase whitefly vector activity that transmits **Tomato Yellow Leaf Curl Virus**. Monitor whitefly populations with sticky traps."
    else:
        analysis_text = f"The AI detected **{top_1_friendly}** on your {crop_name} leaf. Current weather conditions ({temp:.1f}°C, {humidity}% humidity) should be monitored to limit environmental stress on the crop."

    return analysis_text, risk_level, risk_badge, risk_explanation

def get_disease_preventive_measures(top_1_raw: str) -> tuple[str, list[str]]:
    """
    Step 5 Recommendations: Disease-specific preventive checklist.
    """
    raw_lower = top_1_raw.lower() if top_1_raw else "healthy"

    if "healthy" in raw_lower:
        title = "🌿 Recommended Preventive Measures for Healthy Plants"
        measures = [
            "Continue proper, consistent watering at the plant root base",
            "Maintain adequate sunlight exposure and canopy plant spacing",
            "Maintain soil nutrition, organic compost, and proper drainage",
            "Continue regular visual monitoring of foliage for early symptom detection"
        ]
    elif "bacterial_spot" in raw_lower:
        title = "🛡️ Recommended Preventive Measures for Bacterial Spot"
        measures = [
            "Avoid handling wet plants to prevent bacterial cross-contamination",
            "Remove severely affected plant material using sanitized pruning tools",
            "Improve field sanitation and weed management around the crop",
            "Avoid unnecessary overhead irrigation; apply water directly at the base",
            "Monitor spread to nearby healthy plants after wet weather periods"
        ]
    elif "spider_mites" in raw_lower:
        title = "🛡️ Recommended Preventive Measures for Two-Spotted Spider Mites"
        measures = [
            "Inspect leaf undersides regularly for fine webbing or stippling spots",
            "Monitor for active mite populations during warm, dry weather spells",
            "Maintain plant health and adequate soil hydration to limit stress",
            "Use appropriate control methods according to local agricultural guidance"
        ]
    elif any(f in raw_lower for f in ["early_blight", "late_blight", "septoria", "leaf_mold", "target_spot"]):
        title = "🛡️ Recommended Preventive Measures for Fungal Disease"
        measures = [
            "Improve air circulation around plant foliage through selective pruning",
            "Avoid unnecessary overhead watering to keep foliage dry",
            "Remove severely affected leaves promptly and dispose of safely",
            "Remove infected plant debris from the growing area",
            "Maintain proper soil drainage to avoid waterlogging around roots",
            "Monitor plants closely after rainfall or humid weather"
        ]
    elif any(v in raw_lower for v in ["mosaic_virus", "yellowleaf__curl"]):
        title = "🛡️ Recommended Preventive Measures for Viral Infection"
        measures = [
            "Isolate or remove severely infected virus-affected plants early",
            "Sanitize hands and gardening tools thoroughly after handling plants",
            "Manage insect vectors (such as whiteflies) using yellow sticky traps",
            "Avoid handling foliage when wet to minimize mechanical transmission"
        ]
    else:
        title = "🛡️ Recommended General Preventive Measures"
        measures = [
            "Inspect foliage regularly for changing symptom patterns",
            "Maintain proper soil moisture and root drainage",
            "Ensure adequate air circulation across the crop canopy",
            "Seek local agricultural extension guidance for specific management"
        ]
    return title, measures

# ==========================================
# MAIN APPLICATION LOGIC
# ==========================================
def main():
    # Session State Initialization
    if "user_location" not in st.session_state:
        st.session_state["user_location"] = "Bhopal, India"
    if "weather_info" not in st.session_state:
        st.session_state["weather_info"] = None
    if "geocoded_info" not in st.session_state:
        st.session_state["geocoded_info"] = None
    if "weather_error" not in st.session_state:
        st.session_state["weather_error"] = None
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    # Streamlined Sidebar
    with st.sidebar:
        st.markdown("### 🌿 Plant Disease Detection AI")
        st.markdown("---")
        
        st.markdown("#### 📌 Project Details")
        st.write("**Architecture:** MobileNetV3Large")
        st.write("**Classes:** 15 Trained Classes")
        st.write("**Input:** 224 × 224 RGB")
        
        st.markdown("---")
        st.markdown("#### 🌾 Supported Crops")
        st.write("🌶️ **Pepper (Bell)**")
        st.write("🥔 **Potato**")
        st.write("🍅 **Tomato**")
        
        st.markdown("---")
        st.markdown("#### ⚙️ Model Status")

    # Load trained model with error handling
    try:
        model, loaded_path = load_trained_model()
        st.sidebar.success("✅ Model Loaded")
    except FileNotFoundError as fnf_error:
        st.sidebar.error("❌ Model Missing")
        st.error(f"❌ **Model File Missing:** `{str(fnf_error)}`")
        st.stop()
    except Exception as e:
        st.sidebar.error("❌ Model Load Error")
        st.error(f"❌ **Error Loading Model:** `{str(e)}`")
        st.stop()

    # Main Header
    st.markdown('<div class="main-header">🌿 Plant Disease Detection AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered plant disease classification with local weather analysis and preventive guidance</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-desc">Upload a plant leaf image and provide its location to receive a preliminary AI-based disease prediction, environmental weather analysis, and preventive guidance.</div>', unsafe_allow_html=True)

    # Determine workflow progress state for active pill highlighting
    has_location = bool(st.session_state["weather_info"] is not None)
    has_image = False # updated when file uploader checked

    # Render Compact Workflow Progress Bar
    s1_class = "step-completed" if has_location else "step-active"
    s2_class = "step-pending" # updated below dynamically

    st.markdown(
        f"""
        <div class="workflow-container">
            <span class="step-pill {s1_class}">📍 1. Location</span>
            <span class="step-arrow">→</span>
            <span class="step-pill step-pending" id="pill-step-2">📷 2. Leaf Image</span>
            <span class="step-arrow">→</span>
            <span class="step-pill step-pending" id="pill-step-3">🔬 3. AI Diagnosis</span>
            <span class="step-arrow">→</span>
            <span class="step-pill step-pending" id="pill-step-4">🌦️ 4. Weather Analysis</span>
            <span class="step-arrow">→</span>
            <span class="step-pill step-pending" id="pill-step-5">🌱 5. Prevention</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================
    # STEP 1 — PLANT LOCATION
    # ==========================================
    st.markdown('<div class="step-header">📍 STEP 1 — PLANT LOCATION</div>', unsafe_allow_html=True)
    st.markdown("Enter the location where the plant is growing.")

    with st.form("step1_location_form"):
        loc_col1, loc_col2 = st.columns([3.5, 1.2])
        with loc_col1:
            location_input = st.text_input(
                "Plant Location:",
                value=st.session_state["user_location"],
                placeholder="Bhopal, India",
                label_visibility="collapsed"
            )
        with loc_col2:
            check_weather_btn = st.form_submit_button("🌦️ Check Weather", use_container_width=True)

    # Trigger Weather Check
    if check_weather_btn or (st.session_state["weather_info"] is None and st.session_state["weather_error"] is None):
        if location_input and location_input.strip():
            st.session_state["user_location"] = location_input.strip()
            with st.spinner("Retrieving weather metrics..."):
                success_geo, geo_res = geocode_location(location_input.strip())
                if success_geo:
                    st.session_state["geocoded_info"] = geo_res
                    success_wx, wx_res = get_weather_forecast(geo_res["latitude"], geo_res["longitude"])
                    if success_wx:
                        st.session_state["weather_info"] = wx_res
                        st.session_state["weather_error"] = None
                    else:
                        st.session_state["weather_info"] = None
                        st.session_state["weather_error"] = wx_res
                else:
                    st.session_state["geocoded_info"] = None
                    st.session_state["weather_info"] = None
                    st.session_state["weather_error"] = geo_res
        else:
            st.warning("Please enter your location first.")

    if st.session_state["weather_error"]:
        st.warning(st.session_state["weather_error"])

    # Render Compact Weather Metrics Cards
    if st.session_state["weather_info"] and st.session_state["geocoded_info"]:
        geo = st.session_state["geocoded_info"]
        wx = st.session_state["weather_info"]
        curr = wx["current"]

        w_col1, w_col2, w_col3, w_col4, w_col5 = st.columns(5)
        w_col1.metric("📍 Location", geo["display_name"])
        w_col2.metric("🌡️ Temperature", f"{curr['temp']:.1f} °C")
        w_col3.metric("💧 Humidity", f"{curr['humidity']}%")
        w_col4.metric("🌧️ Rain Prob.", f"{curr['rain_prob']}%")
        w_col5.metric("💨 Wind Speed", f"{curr['wind']:.1f} km/h")

    st.markdown("---")

    # ==========================================
    # STEP 2 — UPLOAD LEAF IMAGE
    # ==========================================
    st.markdown('<div class="step-header">📷 STEP 2 — UPLOAD LEAF IMAGE</div>', unsafe_allow_html=True)
    st.markdown("Upload a clear image of a Pepper, Potato, or Tomato leaf.")

    uploaded_file = st.file_uploader(
        "Upload Leaf Photograph:",
        type=["jpg", "jpeg", "png"],
        help="Best results come from a clear, well-lit image where the leaf is visible.",
        label_visibility="collapsed",
        key=f"leaf_uploader_{st.session_state['uploader_key']}"
    )
    st.caption("💡 *Best results come from a clear, well-lit image where the leaf is visible. Supported formats: JPG, JPEG, PNG.*")

    if uploaded_file is None:
        st.info("👆 Please upload a plant leaf image above to view AI Disease Diagnosis.")
        return

    # Load uploaded image safely
    try:
        raw_image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to load uploaded image: `{str(e)}`")
        st.stop()

    st.markdown("---")

    # ==========================================
    # STEP 3 — AI DIAGNOSIS
    # ==========================================
    st.markdown('<div class="step-header">🔬 STEP 3 — AI DIAGNOSIS</div>', unsafe_allow_html=True)

    diag_col1, diag_col2 = st.columns([1, 1], gap="medium")

    with diag_col1:
        st.subheader("📸 Uploaded Leaf Image")
        st.image(raw_image, caption=f"Selected File: {uploaded_file.name}", use_container_width=True)

    with diag_col2:
        st.subheader("🔍 Prediction Results")

        with st.spinner("Analyzing leaf features with MobileNetV3Large..."):
            rgb_image = raw_image.convert("RGB")
            resized_image = rgb_image.resize((224, 224))
            img_array = np.array(resized_image, dtype=np.float32)
            img_batch = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_batch, verbose=0)
            probabilities = predictions[0]

            top_1_idx = int(np.argmax(probabilities))
            top_1_raw = CLASS_NAMES[top_1_idx]
            top_1_friendly = format_user_friendly_name(top_1_raw)
            top_1_confidence = float(probabilities[top_1_idx]) * 100
            detected_crop = detect_crop_from_class(top_1_raw)

            top_3_indices = np.argsort(probabilities)[::-1][:3]

        is_healthy = "healthy" in top_1_raw.lower()
        card_class = "prediction-card-healthy" if is_healthy else "prediction-card-diseased"
        badge_icon = "🌿" if is_healthy else "🦠"

        # Confidence level badges (>=80% High, >=60% Moderate, <60% Low)
        if top_1_confidence >= 80.0:
            conf_badge = '<span style="background-color: #C8E6C9; color: #1B5E20; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.88rem;">🟢 High Confidence</span>'
            conf_level_label = "High Confidence"
        elif top_1_confidence >= 60.0:
            conf_badge = '<span style="background-color: #FFF9C4; color: #F57F17; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.88rem;">🟡 Moderate Confidence</span>'
            conf_level_label = "Moderate Confidence"
        else:
            conf_badge = '<span style="background-color: #FFCDD2; color: #B71C1C; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.88rem;">🔴 Low Confidence</span>'
            conf_level_label = "Low Confidence"

        st.markdown(
            f"""
            <div class="{card_class}">
                <div class="metric-label">{badge_icon} Predicted Condition</div>
                <div class="metric-value">{top_1_friendly}</div>
                <div class="confidence-text">
                    <strong style="color: #212529;">Confidence:</strong> 
                    <span style="color: #111111; font-weight: 700;">{top_1_confidence:.2f}%</span>
                    &nbsp; {conf_badge}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if top_1_confidence < 60.0:
            st.warning("⚠️ The model has low confidence in this diagnosis. Consider uploading a clearer leaf photograph.")

        # Top 3 Predictions
        st.markdown("#### 📊 Top 3 Predictions")
        medals = ["🥇", "🥈", "🥉"]
        for rank, idx in enumerate(top_3_indices, start=1):
            raw_cls = CLASS_NAMES[idx]
            friendly_cls = format_user_friendly_name(raw_cls)
            conf = float(probabilities[idx]) * 100
            medal_icon = medals[rank - 1]
            st.write(f"**{medal_icon} {friendly_cls}** — `{conf:.2f}%`")
            st.progress(min(float(conf / 100.0), 1.0))

        top_3_chart_df = pd.DataFrame({
            "Condition": [format_user_friendly_name(CLASS_NAMES[i]) for i in top_3_indices[::-1]],
            "Confidence (%)": [float(probabilities[i]) * 100 for i in top_3_indices[::-1]]
        }).set_index("Condition")
        st.bar_chart(top_3_chart_df, horizontal=True)

        with st.expander("📈 Detailed Breakdown (All 15 Classes)"):
            for idx, raw_cls in enumerate(CLASS_NAMES):
                friendly_cls = format_user_friendly_name(raw_cls)
                conf = float(probabilities[idx]) * 100
                st.write(f"**{friendly_cls}:** `{conf:.2f}%`")

    st.markdown("---")

    # ==========================================
    # STEP 4 — DISEASE + WEATHER ANALYSIS
    # ==========================================
    st.markdown('<div class="step-header">🌦️ STEP 4 — DISEASE + WEATHER ANALYSIS</div>', unsafe_allow_html=True)

    curr_wx = st.session_state["weather_info"]["current"] if st.session_state["weather_info"] else None

    analysis_text, risk_level, risk_badge, risk_explanation = analyze_disease_and_weather(
        crop_name=detected_crop,
        top_1_raw=top_1_raw,
        top_1_friendly=top_1_friendly,
        current_weather=curr_wx
    )

    st.markdown(
        f"""
        <div class="analysis-card">
            <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 8px;">
                💡 Disease-Specific Environmental Analysis:
            </div>
            <div>
                {analysis_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Prominent Weather-Based Monitoring Level Card
    st.markdown(
        f"""
        <div class="monitoring-card">
            <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 6px;">
                Weather-Based Monitoring Level: {risk_badge}
            </div>
            <div style="font-size: 0.94rem; color: #424242;">
                {risk_explanation}
            </div>
            <div style="font-size: 0.86rem; color: #616161; margin-top: 6px; font-style: italic;">
                *Note: Environmental insights indicate weather pressure favorability and do not diagnose disease occurrence.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ==========================================
    # STEP 5 — PREVENTIVE MEASURES
    # ==========================================
    st.markdown('<div class="step-header">🌱 STEP 5 — PREVENTIVE MEASURES</div>', unsafe_allow_html=True)

    prev_title, prev_measures = get_disease_preventive_measures(top_1_raw)
    st.markdown(f"#### {prev_title}")

    measures_html = "".join([f'<div class="preventive-item">✓ {m}</div>' for m in prev_measures])
    st.markdown(
        f"""
        <div class="preventive-card">
            {measures_html}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Educational Disclaimer
    st.markdown(
        """
        <div class="disclaimer-card">
            ⚠️ <strong>Disclaimer:</strong> AI predictions are preliminary and intended for educational/informational purposes. Weather-based insights are general environmental guidance and should not replace professional agricultural diagnosis.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ==========================================
    # FINAL PLANT HEALTH SUMMARY
    # ==========================================
    st.markdown("### 📋 PLANT HEALTH SUMMARY")

    location_name = st.session_state["geocoded_info"]["display_name"] if st.session_state["geocoded_info"] else (st.session_state["user_location"] if st.session_state["user_location"] else "Not specified")
    
    if curr_wx:
        wx_desc = "Humid / Rain Risk" if curr_wx['humidity'] >= 70 or curr_wx['rain_prob'] >= 50 else f"{curr_wx['temp']:.1f}°C, {curr_wx['humidity']}% Humidity"
    else:
        wx_desc = "Unavailable"

    if "healthy" in top_1_raw.lower():
        final_rec = f"Your {detected_crop} foliage appears healthy. Continue proper base watering, soil drainage, and regular leaf inspection."
    else:
        final_rec = f"Inspect your {detected_crop} foliage regularly for {top_1_friendly} symptoms. Improve canopy air circulation, avoid overhead foliage wetness, and consult local extension officers if symptoms worsen."

    st.markdown(
        f"""
        <div class="summary-card">
            <div style="font-size: 1.05rem; line-height: 1.9; color: #1B5E20;">
                • <strong>Location:</strong> {location_name}<br>
                • <strong>Detected Condition:</strong> {top_1_friendly}<br>
                • <strong>AI Confidence:</strong> {top_1_confidence:.2f}% ({conf_level_label})<br>
                • <strong>Weather:</strong> {wx_desc}<br>
                • <strong>Monitoring:</strong> {risk_level}<br>
                • <strong>Recommended Action:</strong> {final_rec}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    def reset_analysis():
        st.session_state["uploader_key"] += 1

    # Prominent Reset Action Button
    st.button("🔄 Analyze Another Image", use_container_width=True, on_click=reset_analysis)

# Run Streamlit Application
if __name__ == "__main__":
    main()
