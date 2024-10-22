from flask import Flask, request, jsonify, render_template
from flask_pymongo import PyMongo
from transformers import AutoTokenizer, AutoModelForCausalLM
import datetime
import joblib  # Assuming you are using a library like joblib for loading your model
import pandas as pd
import re
import nltk

app = Flask(__name__)

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/llm_database"
mongo = PyMongo(app)

# Load the LLM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('model_dir/tokenizer')
model = AutoModelForCausalLM.from_pretrained('model_dir/model')

# Load your trained skill prediction model and vectorizer
skill_model = joblib.load('model_dir/model/kmeans_model.joblib')
vectorizer = joblib.load('model_dir/model/vectorizer.joblib')

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# List of skills corresponding to KMeans clusters (adjust as necessary)
skills_mapping = [
    "Python",
    "Data Analysis",
    "Machine Learning",
    "Deep Learning",
    "Data Visualization",
    "Statistics",
    "SQL",
    "Big Data",
    "Cloud Computing",
    "Web Development",
    # Add other skills corresponding to each index
]

def predict_skills(job_title):
    # Preprocess the input job title
    processed_title = preprocess_text(job_title)
    
    # Vectorize the processed job title using the same vectorizer
    vectorized_title = vectorizer.transform([processed_title])
    
    # Predict skills using the KMeans model
    predicted_skills_indices = skill_model.predict(vectorized_title)
    
    # Convert predicted skills indices to a list
    predicted_skills_indices = predicted_skills_indices.tolist()  # Converts to a list if it's an array
    
    # Map indices to skill names
    predicted_skills = [skills_mapping[i] for i in predicted_skills_indices if i < len(skills_mapping)]
    
    return predicted_skills if predicted_skills else ["No skills predicted."]

def generate_topics(text, max_input_length=50, max_new_tokens=20):
    inputs = tokenizer.encode(text[:max_input_length], return_tensors='pt')
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Homepage route
@app.route('/')
def index():
    return render_template('index.html')

def is_valid_input(text):
    # Define a simple validation rule (you can adjust this)
    # Example: Check if the input has at least two words and contains letters
    return len(text.split()) > 1 and any(char.isalpha() for char in text)


# Process the input text with the LLM and store results in MongoDB
@app.route('/process', methods=['POST'])
def process_text():
    data = request.json
    input_text = data['text']
    prediction_type = data['type']
    
    if not is_valid_input(input_text):
        return jsonify({'error': 'Invalid input. Please provide a valid job title.'})

    if prediction_type == '1':
        # Generate topics
        generated_text = generate_topics(input_text)
    elif prediction_type == '2':
        # Predict skills
        generated_text = predict_skills(input_text)
    else:
        return jsonify({'error': 'Invalid prediction type'})

    # Store the input and output in the MongoDB database
    record = {
        'input_text': input_text,
        'processed_text': str(generated_text),  # Ensure it's a string
        'timestamp': datetime.datetime.utcnow()
    }
    mongo.db.processed_texts.insert_one(record)

    # Return the processed text as JSON response
    return jsonify({'processed_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
