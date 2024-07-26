from src.logger import logging
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.inference_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age=request.form.get('age'),
            gender=request.form.get('gender'),
            income=request.form.get('income'),
            campaign_channel=request.form.get('campaign_channel'),
            campaign_type=request.form.get('campaign_type'),
            ad_spend=request.form.get('ad_spend'),
            click_through_rate=request.form.get('click_through_rate'),
            conversion_rate=request.form.get('conversion_rate'),
            website_visits=request.form.get('website_visits'),
            pages_per_visit=request.form.get('pages_per_visit'),
            time_on_site=request.form.get('time_on_site'),
            social_shares=request.form.get('social_shares'),
            email_opens=request.form.get('email_opens'),
            email_clicks=request.form.get('email_clicks'),
            previous_purchases=request.form.get('previous_purchases'),
            loyalty_points=request.form.get('loyalty_points')
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('index.html', results=results[0])


if __name__ == '__main__':
    app.run()