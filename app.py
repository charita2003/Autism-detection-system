from flask import Flask, request, render_template, redirect, url_for, session, flash
import numpy as np
import joblib
import tensorflow as tf
import skfuzzy.control as ctrl
import os
import pandas as pd
import json
from tensorflow.keras.preprocessing import image
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
dt_model = joblib.load("decision_tree_model.pkl")
cnn_model = tf.keras.models.load_model("asd_cnn_model.h5")

USER_FILE = 'users.json'

def load_users():
    try:
        with open(USER_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open(USER_FILE, 'w') as f:
        json.dump(users, f)

@app.route('/register', methods=['GET', 'POST'])
def register():
    users = load_users()
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        number = request.form['number']
        address = request.form['address']

        if username in users:
            flash('User already exists! Please log in.', 'danger')
            return redirect(url_for('login'))

        users[username] = {
            'email': email,
            'password': generate_password_hash(password),
            'number': number,
            'address': address
        }
        save_users(users)
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    users = load_users()
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username]['password'], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

class_names = ['Adolescents ASD', 'Adolescents healthy', 'Adult ASD', 'Adult healthy', 'Children ASD', 'Children_healthy']

def predict_csv_model(features):
    input_df = pd.DataFrame([features], columns=[
        'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
        'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
        'age', 'gender', 'jundice', 'used_app_before', 'result', 'age_desc'
    ])
    pred_prob = dt_model.predict_proba(input_df)[0][1]
    pred_label = dt_model.predict(input_df)[0]
    return pred_prob, pred_label

def predict_image_model(img_path):
    test_image = image.load_img(img_path, target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0) / 255.0
    prediction = cnn_model.predict(test_image)
    pred_label = np.argmax(prediction, axis=1)[0]
    return class_names[pred_label]

# Fuzzy Logic Setup
probability = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'probability')
severity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'severity')
final_decision = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'final_decision')

probability.automf(names=['very_low', 'low', 'medium', 'high', 'very_high'])
severity.automf(names=['very_low', 'low', 'medium', 'high', 'very_high'])
final_decision.automf(names=['very_low', 'low', 'medium', 'high', 'very_high'])

rules = [
    ctrl.Rule(probability['very_low'] | severity['very_low'], final_decision['very_low']),
    ctrl.Rule(probability['low'] | severity['low'], final_decision['low']),
    ctrl.Rule(probability['medium'] & severity['medium'], final_decision['medium']),
    ctrl.Rule(probability['high'] & severity['high'], final_decision['high']),
    ctrl.Rule(probability['very_high'] & severity['very_high'], final_decision['very_high'])
]
control_system = ctrl.ControlSystem(rules)
fuzzy_decision = ctrl.ControlSystemSimulation(control_system)

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            feature_values = [
                int(request.form['A1_Score']), int(request.form['A2_Score']),
                int(request.form['A3_Score']), int(request.form['A4_Score']),
                int(request.form['A5_Score']), int(request.form['A6_Score']),
                int(request.form['A7_Score']), int(request.form['A8_Score']),
                int(request.form['A9_Score']), int(request.form['A10_Score']),
                int(request.form['age']), int(request.form['gender']),
                int(request.form['jundice']), int(request.form['used_app_before']),
                int(request.form['result']), int(request.form['age_desc'])
            ]
        except ValueError:
            flash("Invalid input values. Please ensure all fields are correctly filled.", "danger")
            return redirect(url_for('index'))

        risk_score = 0
        for i in range(1, 11):
            value = request.form.get(f'A{i}_Score')
            if value and value.isdigit():
                risk_score += int(value)
        risk_score = risk_score / 10.0

        pred_prob, pred_label = predict_csv_model(feature_values)
        if pred_prob == 0.0: pred_prob = 0.
        elif pred_prob == 1.0: pred_prob = 0.7

        image_result = "No image provided"
        if 'image' in request.files and request.files['image'].filename != '':
            img = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
            img.save(img_path)
            image_result = predict_image_model(img_path)

        severity_value = 0.7 if 'ASD' in image_result else 0.3
        fuzzy_decision.input['probability'] = pred_prob
        fuzzy_decision.input['severity'] = severity_value
        fuzzy_decision.compute()
        # final_score = fuzzy_decision.output['final_decision']
        final_score = (pred_prob + severity_value) / 2


        # Updated strict decision: only Diseased if both > 0.7
        final_diagnosis = "Autism Detected" if pred_prob >= 0.7 and severity_value >= 0.7 else "Autism Not Detected"

        return render_template("result.html",
                               pred_prob=round(pred_prob, 2),
                               pred_label="Autism Detected" if pred_label == 1 else "No Autism",
                               image_result=image_result,
                               final_result=f"{final_diagnosis} (Score: {round(final_score, 2)})",
                               risk_score=round(risk_score, 2))

    return render_template("index.html")

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out successfully.", "success")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
