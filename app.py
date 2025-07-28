import os
from flask import Flask, render_template, request, jsonify, session
import joblib
import numpy as np
from PIL import Image
import google.generativeai as genai
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from werkzeug.utils import secure_filename
import datetime
import base64
import io

from dotenv import load_dotenv
import google.api_core.exceptions

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv('FLASK_SECRET_KEY', 'a_super_secret_key_that_you_should_change_in_production')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. "
                     "Please set it in your .env file or system environment.")

genai.configure(api_key=GEMINI_API_KEY)

try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini model initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    model = None

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

crop_model = None
disease_model = None

try:
   
    crop_model_path = os.path.join(os.path.dirname(__file__), 'models', 'crop_prediction_model.pkl')
    crop_model = joblib.load(crop_model_path)
    print(f"Crop prediction model loaded successfully from {crop_model_path}")

    disease_model_path = os.path.join(os.path.dirname(__file__), 'models', 'plant_disease_model.h5')
    disease_model = load_model(disease_model_path)
    print(f"Disease prediction model loaded successfully from {disease_model_path}")

except Exception as e:
    print(f"Error loading models: {e}")
    crop_model = None
    disease_model = None

# --- Configure Upload Folder ---
app.config['UPLOAD_FOLDER'] = 'static/uploads' 
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 

disease_solutions = {
    'Potato___healthy': {
        "description": "Your potato plant is healthy and thriving. No signs of disease or pests detected.",
        "common_symptoms": ["Vigorous, green foliage", "Strong stems", "No visible spots, lesions, or wilting"],
        "immediate_actions": ["No immediate action required. Continue routine care."],
        "prevention_tips": ["Maintain proper soil moisture and nutrient balance.", "Ensure good air circulation.", "Regularly monitor for early signs of issues."],
        "dawai": "No specific medicine or pesticide needed.",
        "fertilizer": "Continue with your regular balanced fertilization schedule (e.g., NPK 10:26:26 or 12:32:16) as per potato growth stage for optimal yield.",
        "general_advice": "Maintain good agricultural practices: proper irrigation, regular weeding, and timely monitoring for any new symptoms. Ensure good soil drainage.",
        "confidence": 99,
        "severity": "None"
    },
    'Tomato__Tomato_mosaic_virus': {
        "description": "Tomato Mosaic Virus (ToMV) is a highly contagious viral disease affecting tomato plants, causing distinctive mottling and distortion.",
        "common_symptoms": ["Light green and dark green mottling or mosaic pattern on leaves", "Leaf distortion, puckering, or 'shoestring' appearance", "Stunted plant growth", "Reduced and deformed fruit"],
        "immediate_actions": ["Immediately remove and destroy (burn or deep bury) all infected plants.", "Thoroughly sanitize hands, tools, and stakes after handling infected plants.", "Control aphid and whitefly populations (vectors) using sticky traps or appropriate insecticides."],
        "prevention_tips": ["Use certified virus-free seeds or resistant varieties.", "Avoid handling tobacco products before working with plants.", "Practice strict sanitation: wash hands, disinfect tools.", "Control weeds in and around the field.", "Implement crop rotation."],
        "dawai": "No chemical 'dawai' can cure viral diseases. Focus on prevention and vector control. Use yellow sticky traps to control aphids and whiteflies which spread the virus. Consider applying neem oil as a repellent.",
        "fertilizer": "Ensure balanced nutrition to strengthen plant vigor and natural resistance. Avoid excessive nitrogen, which can lead to lush growth and attract vectors.",
        "general_advice": "Remove and immediately destroy (burn or bury deep) all infected plants to prevent spread. Sanitize hands and tools frequently. Plant resistant varieties if available. Control weeds around the field.",
        "confidence": 92,
        "severity": "High"
    },
    'Tomato_Late_blight': {
        "description": "Late blight is a destructive fungal disease of tomato, especially severe in cool, wet conditions, leading to rapid foliage and fruit decay.",
        "common_symptoms": ["Large, irregular, water-soaked spots on leaves that turn brown/black", "White, fuzzy fungal growth on the underside of leaves (especially in humid conditions)", "Dark, sunken lesions on stems", "Hard, brown, rot on fruits"],
        "immediate_actions": ["Immediately remove and destroy all infected plant parts.", "Apply systemic fungicides containing active ingredients like Metalaxyl or Cymoxanil, combined with a protectant like Mancozeb or Chlorothalonil.", "Improve air circulation around plants by pruning excess foliage."],
        "prevention_tips": ["Use disease-resistant tomato varieties if available.", "Avoid overhead irrigation; water early in the morning at the base of plants.", "Ensure proper plant spacing for good air circulation.", "Practice strict sanitation: remove and destroy all plant debris after harvest.", "Implement crop rotation."],
        "dawai": "Apply systemic fungicides like Metalaxyl + Mancozeb (e.g., Ridomil Gold), Propineb, or Cymoxanil + Mancozeb at the first sign of symptoms, and continue at 7-10 day intervals, especially during cool, humid weather. Rotate fungicides to prevent resistance.",
        "fertilizer": "Ensure adequate potassium (K) fertilization, as it enhances plant resistance. Avoid excessive nitrogen application, which can make plants more susceptible. Calcium is also important for fruit quality.",
        "general_advice": "Improve air circulation by proper plant spacing and pruning lower leaves. Avoid overhead irrigation, water early in the morning. Use resistant varieties. Destroy volunteer potato and tomato plants and crop debris.",
        "confidence": 95,
        "severity": "High"
    },
    'Tomato_Leaf_Mold': {
        "description": "Leaf mold is a fungal disease of tomato, particularly common in humid conditions, affecting leaves and sometimes fruit.",
        "common_symptoms": ["Yellowish-green spots on the upper leaf surface, turning pale brown.", "Olive-green to brownish, velvety fungal growth on the underside of leaves.", "Leaves eventually curl, wilt, and die."],
        "immediate_actions": ["Increase air circulation by pruning lower leaves and side shoots.", "Reduce humidity in greenhouses or protected cultivation.", "Apply fungicides containing Chlorothalonil, Mancozeb, or copper-based compounds."],
        "prevention_tips": ["Use resistant tomato varieties.", "Improve ventilation in enclosed growing areas.", "Avoid excessive leaf wetness; water at the base of plants.", "Remove and destroy infected plant debris."],
        "dawai": "Fungicides containing Chlorothalonil or Mancozeb can be effective. Apply as per label instructions, especially during periods of high humidity. Systemic fungicides like Azoxystrobin can also be used.",
        "fertilizer": "Maintain balanced nutrition. Ensure adequate calcium and potassium, which help in overall plant health and resistance.",
        "general_advice": "Increase air circulation by pruning and wider plant spacing. Reduce humidity in protected cultivation (polyhouse/net house) through proper ventilation. Avoid prolonged leaf wetness. Remove infected leaves.",
        "confidence": 88,
        "severity": "Medium"
    },
    'Tomato_healthy': {
        "description": "Your tomato plant is healthy and growing well. No signs of disease or pests detected.",
        "common_symptoms": ["Lush green leaves", "Strong stem development", "Abundant flowering/fruiting", "No visible abnormalities"],
        "immediate_actions": ["Continue routine watering and feeding.", "Ensure adequate sunlight."],
        "prevention_tips": ["Monitor regularly for pests and diseases.", "Provide proper support (staking).", "Maintain good soil health."],
        "dawai": "No specific medicine or pesticide needed.",
        "fertilizer": "Continue with your regular balanced fertilization schedule (e.g., NPK 19:19:19 or as per soil test) tailored to tomato's growth stage for healthy fruit development.",
        "general_advice": "Maintain optimal irrigation, weed control, and provide staking/support. Regularly monitor plants for early signs of stress or pest activity.",
        "confidence": 98,
        "severity": "None"
    },
    'Tomato__Target_Spot': {
        "description": "Target spot is a fungal disease of tomato characterized by circular spots with concentric rings on leaves, stems, and fruits.",
        "common_symptoms": ["Small, circular, dark brown spots on older leaves, often with a yellow halo.", "Spots enlarge and develop concentric rings (target-like appearance).", "Lesions may appear on stems and fruits.", "Premature defoliation of lower leaves."],
        "immediate_actions": ["Remove and destroy infected plant debris.", "Apply fungicides such as Mancozeb, Chlorothalonil, or those containing strobilurins (e.g., Azoxystrobin) at recommended intervals."],
        "prevention_tips": ["Practice crop rotation (avoid planting solanaceous crops in the same spot for several years).", "Ensure good air circulation by proper spacing and pruning.", "Avoid overhead irrigation, water at the base of plants.", "Use disease-free seeds or transplants."],
        "dawai": "Apply fungicides like Mancozeb, Chlorothalonil, or Azoxystrobin + Difenoconazole at the first sign of disease. Follow spray schedules (e.g., every 7-10 days) especially in humid conditions.",
        "fertilizer": "Ensure adequate potassium and calcium. Balanced nutrition helps overall plant strength against diseases.",
        "general_advice": "Remove and destroy infected plant debris. Improve air circulation. Avoid overhead irrigation. Practice crop rotation (avoid planting tomato, potato, or related crops in the same spot for at least 3 years).",
        "confidence": 89,
        "severity": "Medium"
    },
    'Tomato_Septoria_leaf_spot': {
        "description": "Septoria leaf spot is a common fungal disease of tomato that causes numerous small, dark spots on lower leaves.",
        "common_symptoms": ["Numerous small, circular spots (1/8 to 1/4 inch) on older leaves, with a dark brown border and tan to gray center.", "Tiny black dots (fruiting bodies) visible within the spots.", "Leaves turn yellow, then brown, and fall off prematurely."],
        "immediate_actions": ["Remove and destroy infected lower leaves immediately.", "Apply fungicides containing Chlorothalonil, Mancozeb, or copper-based compounds.", "Avoid working with plants when they are wet."],
        "prevention_tips": ["Clean up all plant debris at the end of the growing season.", "Avoid overhead watering; use drip irrigation or water at the base.", "Ensure good air circulation through proper plant spacing.", "Sanitize tools regularly."],
        "dawai": "Fungicides containing Chlorothalonil, Mancozeb, or copper-based compounds are effective. Apply protectant sprays when conditions are favorable for disease development (warm and wet).",
        "fertilizer": "Maintain general plant health with a balanced fertilizer program. Healthy plants are better able to withstand disease pressure.",
        "general_advice": "Clean up plant debris at the end of the season. Avoid splashing water, as spores spread through water. Improve air circulation. Avoid planting tomatoes too densely. Sanitize tools.",
        "confidence": 91,
        "severity": "Medium"
    },
    'Pepper__bell___Bacterial_spot': {
        "description": "Bacterial spot is a widespread bacterial disease affecting bell peppers, causing lesions on leaves, stems, and fruit.",
        "common_symptoms": ["Small, water-soaked, dark spots on leaves that later become angular and necrotic with yellow halos.", "Spots on fruit are raised, scab-like, and may have a watersoaked margin.", "Yellowing and defoliation of leaves.", "Lesions on stems."],
        "immediate_actions": ["Remove and destroy severely infected plants or plant parts.", "Apply copper-based bactericides combined with Mancozeb.", "Avoid working in the field when plants are wet to prevent bacterial spread."],
        "prevention_tips": ["Use certified disease-free seeds or transplants.", "Practice strict sanitation of tools and equipment.", "Implement a 2-3 year crop rotation, avoiding susceptible crops (peppers, tomatoes, potatoes, eggplant).", "Avoid overhead irrigation."],
        "dawai": "Use copper-based bactericides (e.g., Copper Oxychloride) mixed with Mancozeb for better efficacy. Apply preventively, especially if bacterial spot has been an issue previously. Note: Bactericides prevent spread, they don't cure existing spots.",
        "fertilizer": "Avoid excessive nitrogen, which can lead to tender growth more susceptible to bacterial infections. Ensure adequate calcium and potassium for stronger cell walls.",
        "general_advice": "Use disease-free certified seeds/transplants. Avoid working in the field when plants are wet. Sanitize tools. Practice crop rotation (avoid peppers, tomatoes, potatoes, eggplant for 2-3 years in the same spot).",
        "confidence": 87,
        "severity": "High"
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        "description": "Tomato Yellow Leaf Curl Virus (TYLCV) is a destructive viral disease transmitted by whiteflies, causing severe yellowing and curling of leaves.",
        "common_symptoms": ["Upward and inward curling of leaves, often becoming leathery.", "Severe yellowing (chlorosis) of leaf margins and interveinal areas.", "Stunted plant growth and reduced internode length.", "Small, deformed flowers and fruits, or no fruit production."],
        "immediate_actions": ["Immediately remove and destroy all symptomatic plants.", "Aggressively control whitefly populations using yellow sticky traps and appropriate insecticides (e.g., Imidacloprid, Thiamethoxam).", "Use reflective mulches to deter whiteflies."],
        "prevention_tips": ["Plant resistant or tolerant tomato varieties.", "Use insect-proof netting in protected cultivation.", "Ensure proper weed control, as weeds can harbor whiteflies and the virus.", "Isolate new plantings from older ones.", "Inspect transplants carefully before planting."],
        "dawai": "No chemical cure for the virus itself. The primary focus is whitefly control, as they are the vectors. Use yellow sticky traps. Apply insecticides like Imidacloprid (systemic) or Thiamethoxam to control whitefly populations. Consider reflective mulches to deter whiteflies.",
        "fertilizer": "Ensure plants are well-nourished to increase their resilience. A balanced NPK fertilizer and micronutrients are beneficial.",
        "general_advice": "Remove and destroy all symptomatic plants immediately. Control weeds, as they can harbor whiteflies. Use virus-resistant varieties. Isolate new plantings from older ones. Cover new plants with insect-proof netting if possible.",
        "confidence": 93,
        "severity": "High"
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        "description": "Two-spotted spider mites are tiny arachnids that feed on tomato plant sap, causing stippling and webbing on leaves.",
        "common_symptoms": ["Tiny yellow or white stippling (pinprick dots) on the upper surface of leaves.", "Fine webbing on the undersides of leaves, stems, or between leaves.", "Leaves may turn bronze or yellow, and eventually drop.", "Reduced plant vigor."],
        "immediate_actions": ["Spray affected plants thoroughly with strong jets of water to dislodge mites (especially undersides of leaves).", "Apply insecticidal soap, horticultural oil, or neem oil solution, ensuring full coverage.", "For severe infestations, consider using specific miticides like Abamectin or Spiromesifen."],
        "prevention_tips": ["Maintain good plant hygiene and remove weeds.", "Avoid excessive nitrogen fertilization, which can promote tender growth favored by mites.", "Increase humidity around plants by misting (if fungal issues are not a concern).", "Introduce natural predators like predatory mites if suitable for your system."],
        "dawai": "Spray with horticultural oil, insecticidal soap, or neem oil solution, focusing on the undersides of leaves. For severe infestations, use specific miticides like Abamectin or Spiromesifen. Rotate products to prevent resistance.",
        "fertilizer": "Avoid excessive nitrogen, which can promote tender growth preferred by mites. Healthy, well-nourished plants are more tolerant.",
        "general_advice": "Maintain good plant hygiene. Regularly check undersides of leaves. Increase humidity around plants by misting (if not promoting fungal issues). Introduce predatory mites if suitable for your farming system.",
        "confidence": 85,
        "severity": "Medium"
    },
    'Tomato_Bacterial_spot': { # This is treated as distinct from Pepper bell bacterial spot in your labels.
        "description": "Bacterial spot is a significant bacterial disease of tomato plants, causing irregular lesions on foliage and fruit.",
        "common_symptoms": ["Small, dark, water-soaked spots on leaves that become angular and dark brown with a yellow halo.", "Spots on fruit are small, raised, brown, and scab-like.", "Leaves may turn yellow and drop prematurely.", "Lesions can occur on stems."],
        "immediate_actions": ["Remove and destroy infected plant material.", "Apply copper-based bactericides (e.g., Copper Oxychloride) often mixed with Mancozeb. Apply preventively during favorable conditions."],
        "prevention_tips": ["Use certified disease-free seeds or transplants.", "Avoid overhead irrigation; water plants at the base.", "Practice good field sanitation and crop rotation (avoid solanaceous crops for at least 2-3 years).", "Avoid working in the field when plants are wet."],
        "dawai": "Use copper-based bactericides (e.g., Copper Oxychloride) mixed with Mancozeb for better efficacy. Apply preventively. Note: Bactericides prevent spread, they don't cure existing spots.",
        "fertilizer": "Avoid excessive nitrogen, which can lead to tender growth more susceptible to bacterial infections. Ensure adequate calcium and potassium for stronger cell walls.",
        "general_advice": "Use disease-free certified seeds/transplants. Avoid working in the field when plants are wet. Sanitize tools. Practice crop rotation (avoid peppers, tomatoes, potatoes, eggplant for 2-3 years in the same spot).",
        "confidence": 86,
        "severity": "Medium"
    },
    'Pepper__bell___healthy': {
        "description": "Your bell pepper plant is healthy and showing no signs of disease or pest infestations.",
        "common_symptoms": ["Lush, green leaves", "Robust stem and root system", "Good fruit development", "Absence of spots, discoloration, or wilting"],
        "immediate_actions": ["Continue with regular care and monitoring."],
        "prevention_tips": ["Ensure consistent watering and proper drainage.", "Provide adequate sunlight.", "Regularly inspect plants for any changes."],
        "dawai": "No specific medicine or pesticide needed.",
        "fertilizer": "Continue with a balanced NPK fertilizer program suitable for bell peppers to support vigorous growth and fruit development.",
        "general_advice": "Ensure adequate water, sunlight, and proper air circulation. Monitor regularly for any changes in plant health.",
        "confidence": 99,
        "severity": "None"
    },
    'Tomato_Early_blight': {
        "description": "Early blight is a common fungal disease of tomato that typically affects older leaves first, causing characteristic bullseye-like spots.",
        "common_symptoms": ["Dark, concentric rings (bullseye pattern) on older leaves.", "Spots often surrounded by a yellow halo.", "Lesions may appear on stems and fruit at the stem end.", "Premature defoliation of lower leaves."],
        "immediate_actions": ["Remove and destroy infected lower leaves and plant debris.", "Apply fungicides containing Chlorothalonil, Mancozeb, or copper-based compounds at the first sign of symptoms, especially during warm, humid weather."],
        "prevention_tips": ["Practice good sanitation: remove all plant debris at the end of the season.", "Rotate crops to non-solanaceous plants for at least 2 years.", "Avoid overhead irrigation; water at the soil line.", "Ensure adequate plant spacing for good air circulation.", "Use resistant varieties."],
        "dawai": "Apply fungicides like Mancozeb, Chlorothalonil, or Azoxystrobin at the first appearance of symptoms, especially during warm, humid periods. Repeat sprays at recommended intervals.",
        "fertilizer": "Maintain balanced soil fertility, especially adequate nitrogen and potassium. Healthy plants can better tolerate early blight infections.",
        "general_advice": "Remove infected lower leaves and plant debris. Practice crop rotation. Ensure good air circulation. Avoid overhead irrigation or water early in the morning to allow leaves to dry.",
        "confidence": 90,
        "severity": "Medium"
    },
    'Potato___Late_blight': {
        "description": "Late blight is a highly destructive fungal disease of potato, thriving in cool, wet conditions and causing rapid blighting of foliage and tubers.",
        "common_symptoms": ["Large, irregular, dark brown to black, water-soaked spots on leaves and stems.", "White, fuzzy fungal growth on the undersides of leaves in humid conditions.", "Rapid wilting and collapse of entire plants.", "Reddish-brown, firm rot on potato tubers, extending into the potato flesh."],
        "immediate_actions": ["Immediately remove and destroy all infected potato plants and tubers.", "Apply systemic fungicides containing Metalaxyl or Cymoxanil, along with a protectant like Mancozeb or Chlorothalonil.", "Hilling up soil around plants can protect tubers from spores."],
        "prevention_tips": ["Use certified disease-free seed potatoes.", "Plant resistant potato varieties if available.", "Avoid overhead irrigation; use furrow or drip irrigation.", "Ensure proper plant spacing for good air circulation.", "Destroy volunteer potato and tomato plants.", "Do not store infected tubers."],
        "dawai": "Apply systemic fungicides like Metalaxyl + Mancozeb (e.g., Ridomil Gold), Propineb, or Cymoxanil + Mancozeb at the first sign of symptoms, and continue at 7-10 day intervals, especially during cool, humid weather. Rotate fungicides to prevent resistance.",
        "fertilizer": "Ensure adequate potassium (K) fertilization, as it enhances plant resistance. Avoid excessive nitrogen application, which can make plants more susceptible. Calcium is also important for tuber quality.",
        "general_advice": "Improve air circulation by proper plant spacing and hilling. Avoid overhead irrigation, water early in the morning. Use resistant varieties. Destroy volunteer potato and tomato plants and crop debris.",
        "confidence": 96,
        "severity": "High"
    },
    'Potato___Early_blight': {
        "description": "Early blight is a common fungal disease of potato that typically affects older leaves first, causing characteristic bullseye-like spots.",
        "common_symptoms": ["Dark brown to black spots with concentric rings (bullseye pattern) on older potato leaves.", "Spots often surrounded by a yellow halo.", "Lesions may appear on stems and tubers.", "Premature defoliation of lower leaves."],
        "immediate_actions": ["Remove and destroy infected lower leaves and plant debris.", "Apply fungicides containing Chlorothalonil, Mancozeb, or copper-based compounds at the first sign of symptoms.", "Ensure plants are well-watered to reduce stress."],
        "prevention_tips": ["Practice good sanitation: remove all plant debris after harvest.", "Rotate crops to non-solanaceous plants for at least 2 years.", "Avoid overhead irrigation; water at the soil line.", "Ensure adequate plant spacing for good air circulation.", "Use resistant varieties where possible."],
        "dawai": "Apply fungicides like Mancozeb, Chlorothalonil, or Azoxystrobin at the first appearance of symptoms, especially during warm, humid periods. Repeat sprays at recommended intervals.",
        "fertilizer": "Maintain balanced soil fertility, especially adequate nitrogen and potassium. Healthy plants can better tolerate early blight infections.",
        "general_advice": "Remove infected lower leaves and plant debris. Practice crop rotation. Ensure good air circulation. Avoid overhead irrigation or water early in the morning to allow leaves to dry.",
        "confidence": 89,
        "severity": "Medium"
    },
    'Unknown Disease': {
        "description": "The specific disease could not be identified. The symptoms observed might be unusual or less common.",
        "common_symptoms": ["Unusual spots or discoloration", "Unexpected wilting or yellowing", "Symptoms not matching common diseases"],
        "immediate_actions": ["Isolate the affected plant if possible.", "Consult a local agricultural expert or extension office for a precise diagnosis.", "Document symptoms with more photos or detailed observations."],
        "prevention_tips": ["Maintain general plant hygiene.", "Monitor closely for any new or spreading symptoms."],
        "dawai": "The specific medicine or pesticide for this condition is unknown from our database. Please consult a local agricultural expert or extension office for a precise diagnosis.",
        "fertilizer": "Continue with a balanced fertilizer unless specific symptoms indicate a nutrient deficiency. Consider a general plant health booster.",
        "general_advice": "Isolate the affected plant if possible to prevent potential spread. Monitor symptoms closely and document changes (e.g., more photos, description of spread)."
    }
}

disease_labels = [
    'Potato___healthy',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_healthy',
    'Tomato__Target_Spot',
    'Tomato_Septoria_leaf_spot',
    'Pepper__bell___Bacterial_spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Bacterial_spot', 
    'Pepper__bell___healthy',
    'Tomato_Early_blight',
    'Potato___Late_blight',
    'Potato___Early_blight'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crop-predict', methods=['GET', 'POST'])
def crop_predict():
    prediction_result = None
    if request.method == 'POST':
        try:
            required_fields = ['location', 'soil_type', 'rainfall', 'temperature', 'humidity', 'season']
            if not all(request.form.get(field) for field in required_fields):
                raise ValueError("Please fill in all fields")

            rainfall = float(request.form.get('rainfall'))
            temperature = float(request.form.get('temperature'))
            humidity = float(request.form.get('humidity'))

            input_data = pd.DataFrame([[
                request.form.get('location'),
                request.form.get('soil_type'),
                rainfall,
                temperature,
                humidity,
                request.form.get('season')
            ]], columns=['Location', 'Soil Type', 'Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)', 'Season'])

            if crop_model:
                prediction = crop_model.predict(input_data)[0]
                prediction_result = f"The best crop to grow is: {prediction}"
            else:
                prediction_result = "Crop prediction service is currently unavailable. Model not loaded."

        except ValueError as e:
            prediction_result = f"Invalid input: {str(e)}"
        except Exception as e:
            prediction_result = "An error occurred during prediction"
            print(f"Crop prediction error: {e}")

    return render_template('crop_predict.html', prediction_result=prediction_result)


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_predict():
    image_url = None
    disease_identified = None
    confidence = None
    description = None
    common_symptoms = []
    immediate_actions = []
    prevention_tips = []
    solution_dawai = None
    solution_fertilizer = None
    solution_general_advice = None
    severity = None
    error_message = None

    if request.method == 'POST' and 'imageUpload' in request.files:
        file = request.files['imageUpload']
        
        if file.filename == '':
            error_message = "Please select an image to upload."
        else:
            try:
                img = Image.open(file.stream).convert('RGB')
                
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                image_url = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()
                
                if disease_model:
              
                    img_resized = img.resize((128, 128)) 
                    img_array = np.asarray(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0

                    predictions = disease_model.predict(img_array)
                    predicted_class_index = np.argmax(predictions, axis=1)[0]
                    
                    if 0 <= predicted_class_index < len(disease_labels):
                        predicted_label = disease_labels[predicted_class_index]
                        confidence_score = predictions[0][predicted_class_index] * 100

                        disease_identified = predicted_label
                        confidence = f"{confidence_score:.2f}%"

                        solution_data = disease_solutions.get(predicted_label, disease_solutions['Unknown Disease'])
                        description = solution_data.get('description', 'No description available.')
                        common_symptoms = solution_data.get('common_symptoms', ['N/A'])
                        immediate_actions = solution_data.get('immediate_actions', ['N/A'])
                        prevention_tips = solution_data.get('prevention_tips', ['N/A'])
                        solution_dawai = solution_data.get('dawai', 'No specific medicine/pesticide advice.')
                        solution_fertilizer = solution_data.get('fertilizer', 'No specific fertilizer advice.')
                        solution_general_advice = solution_data.get('general_advice', 'No general advice available.')
                        severity = solution_data.get('severity', 'Unknown')
                    else:
                        error_message = "Predicted class index out of bounds. Model output might be unexpected."
                        solution_data = disease_solutions['Unknown Disease']
                        description = solution_data.get('description')
                        common_symptoms = solution_data.get('common_symptoms')
                        immediate_actions = solution_data.get('immediate_actions')
                        prevention_tips = solution_data.get('prevention_tips')
                        solution_dawai = solution_data.get('dawai')
                        solution_fertilizer = solution_data.get('fertilizer')
                        solution_general_advice = solution_data.get('general_advice')
                        severity = solution_data.get('severity')
                        disease_identified = "Unknown Disease" 


                else:
                    error_message = "Disease prediction model not loaded."

            except Exception as e:
                error_message = f"Error processing image or predicting disease: {e}"
                print(f"Disease prediction error: {e}")

    return render_template('disease_detect.html', 
                           image_url=image_url,
                           disease_identified=disease_identified,
                           confidence=confidence,
                           description=description,
                           common_symptoms=common_symptoms,
                           immediate_actions=immediate_actions,
                           prevention_tips=prevention_tips,
                           solution_dawai=solution_dawai,
                           solution_fertilizer=solution_fertilizer,
                           solution_general_advice=solution_general_advice,
                           severity=severity,
                           error_message=error_message)

@app.route('/chat_ai')
def chat_ai():

    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('chat_ai.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"response": "No message received."}), 400


    history = session.get('chat_history', [])

    try:
        if not model:
            return jsonify({"response": "AI model not available on server. Please check server logs for initialization errors."}), 503 # Service Unavailable

        chat_session = model.start_chat(history=history)

        response = chat_session.send_message(user_message, safety_settings=SAFETY_SETTINGS)

        ai_response_text = ""
        try:
            ai_response_text = response.text
        except ValueError:

            print(f"Gemini response did not contain text content. Raw response: {response}")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                ai_response_text = "Your message was blocked due to safety concerns. Please try rephrasing."
            elif response.candidates and len(response.candidates) > 0 and response.candidates[0].finish_reason == 'SAFETY':
                ai_response_text = "The AI's response was blocked due to safety filters."
            else:
                ai_response_text = "The AI did not provide a complete response."


        history.append({'role': 'user', 'parts': [{'text': user_message}]})
        history.append({'role': 'model', 'parts': [{'text': ai_response_text}]})
        session['chat_history'] = history 

        return jsonify({"response": ai_response_text})

    except genai.types.BlockedPromptException as e:
        print(f"Prompt blocked by safety settings: {e}")
        return jsonify({"response": "Your message was blocked by safety filters. Please try rephrasing."}), 400
    except google.api_core.exceptions.GoogleAPIError as e: 
        print(f"Gemini API error (GoogleAPIError): {e}")
        return jsonify({"response": f"Error communicating with AI: {e}. Please try again later."}), 500
    except Exception as e: 
        print(f"An unexpected error occurred in chat route: {e}")
        return jsonify({"response": f"Sorry, an internal error occurred: {e}. Please try again later."}), 500


# if __name__ == '__main__':
    
#     app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000)) 
    app.run(host="0.0.0.0", port=port)