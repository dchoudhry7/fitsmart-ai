from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load trained model
model = joblib.load("models/model.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        duration = int(request.form['duration'])
        heart_rate = int(request.form['heart_rate'])
        body_temp = float(request.form['body_temp'])

        # Prepare data
        data = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])

        # Prediction
        calories = model.predict(data)[0]

        # BMI calculation
        bmi = weight / ((height / 100) ** 2)

        if bmi < 18.5:
            status = "Underweight"
            advice = "Increase calorie intake & strength training"
        elif bmi < 25:
            status = "Normal"
            advice = "Maintain balanced diet & regular exercise"
        elif bmi < 30:
            status = "Overweight"
            advice = "Focus on cardio & calorie deficit"
        else:
            status = "Obese"
            advice = "Strict diet + daily workout needed"

        # 🔥 Save history safely
        new_data = pd.DataFrame([[age, height, weight, duration, calories]],
                                columns=['age', 'height', 'weight', 'duration', 'calories'])

        new_data.to_csv("data/user_history.csv",
                        mode='a',
                        header=False,
                        index=False,
                        lineterminator='\n')

        # 🔥 Read history
        history = pd.read_csv("data/user_history.csv")

        # Sort for better plotting
        history = history.sort_values(by='duration')

        # 🔥 Create SMALL CLEAN GRAPH
        plt.figure(figsize=(4,3))  # smaller size

        plt.scatter(history['duration'], history['calories'])

        plt.xlabel("Duration")
        plt.ylabel("Calories")
        plt.title("Progress")

        plt.tight_layout()

        plt.savefig("static/graph.png")
        plt.close()

        return render_template('result.html',
                               calories=int(calories),
                               bmi=round(bmi, 2),
                               status=status,
                               advice=advice)

    except Exception as e:
        return f"Error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)