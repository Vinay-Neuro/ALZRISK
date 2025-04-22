from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'lightgbm_model_22-04-2025.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    input_values = [float(x) for x in request.form.values()]
    input_array = np.array(input_values).reshape(1, -1)  # Ensure correct shape

    # Predict probability
    probability = model.predict_proba(input_array)[0][1]  # Probability of class 1

    # Format probability as percentage
    prob_percent = round(probability * 100, 2)
    output = f'Estimated risk of developing Alzheimer\'s: {prob_percent}%'

    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)
