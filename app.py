from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    input_text = request.form['text']
    
    # Preprocess and vectorize the input text
    input_vector = vectorizer.transform([input_text])
    
    # Make prediction
    prediction = model.predict(input_vector)
    
    # Map prediction to sentiment label
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    
    return jsonify({'sentiment': sentiment})

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5001, debug=True)
