from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    try:
        reading_score = float(request.form.get('reading_score', ''))
        writing_score = float(request.form.get('writing_score', ''))

        if not (
            0 <= reading_score <= 100
            and 0 <= writing_score <= 100
        ):
            raise ValueError("Scores must be between 0 and 100.")

    except (TypeError, ValueError):
        return render_template(
            'home.html',
            error="Enter valid reading and writing scores between 0 and 100."
        ), 400

    allowed_values = {
        'gender': ['male', 'female'],
        'ethnicity': ['group A', 'group B', 'group C', 'group D', 'group E'],
        'parental_level_of_education': [
            "associate's degree",
            "bachelor's degree",
            "high school",
            "master's degree",
            "some college",
            "some high school",
        ],
        'lunch': ['free/reduced', 'standard'],
        'test_preparation_course': ['none', 'completed'],
    }

    for field, choices in allowed_values.items():
        if request.form.get(field) not in choices:
            return render_template(
                'home.html',
                error="Please select a valid option in every dropdown."
            ), 400

    data = CustomData(
        gender=request.form.get('gender'),
        race_ethnicity=request.form.get('ethnicity'),
        parental_level_of_education=request.form.get(
            'parental_level_of_education'
        ),
        lunch=request.form.get('lunch'),
        test_preparation_course=request.form.get(
            'test_preparation_course'
        ),
        reading_score=reading_score,
        writing_score=writing_score
    )

    pred_df = data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)

    return render_template(
        'home.html',
        results=round(float(results[0]), 2)
    )
    

if __name__=="__main__":
    app.run(host="0.0.0.0") 