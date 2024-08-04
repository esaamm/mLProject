from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__) # Created the entry point for the flask application.

app=application

# Route for a home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():  # This function is called in the home.htlml file in the form action field .
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(   # When request is POST and the data filled in the form comes here in the data object of CustomData class.
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )  # data is an object of class CustomData created.
        
        pred_df=data.get_data_as_data_frame()  # Here all the collected data from the form gets converted into a data frame.
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline() # Object of PredictPipeline class is created.
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        