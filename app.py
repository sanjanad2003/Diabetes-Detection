from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd

app = Flask(__name__)

# Load trained models & scaler
log_reg = pickle.load(open('models_dir/logistic_regression.pkl', 'rb'))
dec_tree = pickle.load(open('models_dir/decision_tree.pkl', 'rb'))
scaler = pickle.load(open('models_dir/scaler.pkl', 'rb'))
metrics = pickle.load(open('models_dir/metrics.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()  # Convert form data to dictionary
    model_choice = form_data.pop('model')  # Extract the selected model

    try:
        data = [float(x) for x in form_data.values()]  # Convert only numeric inputs
    except ValueError:
        return jsonify({'error': 'Invalid input: Please enter numerical values only.'})

    input_data = np.array(data).reshape(1, -1)
    scaled_data = scaler.transform(input_data)

    # Select model based on choice
    if model_choice == 'logistic_regression':
        prediction = log_reg.predict(scaled_data)
    else:
        prediction = dec_tree.predict(scaled_data)

    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    return jsonify({'prediction': result})



@app.route('/metrics')
def show_metrics():
    return jsonify(metrics)

@app.route('/charts')
def charts():
    df = pd.read_csv('dataset/diabetes.csv')

    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df['Glucose'], kde=True, bins=30, ax=ax)
    plt.title("Glucose Level Distribution")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded = base64.b64encode(img.getvalue()).decode()
    
    return render_template('charts.html', chart=encoded)

if __name__ == '__main__':
    app.run(debug=True)
