from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model_path = '/Users/vinaysuresh/Downloads/render/lightgbm_model.pkl'
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

    # Make prediction
    prediction = model.predict(input_array)
    output = 'Risk of developing Alzheimer\'s' if prediction[0] == 1 else 'No risk of developing Alzheimer\'s'

    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)
