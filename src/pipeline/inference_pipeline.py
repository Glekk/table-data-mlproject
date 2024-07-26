import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline():
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'models/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e, sys)
    

class CustomData():
    def __init__(self, 
                 age: int,
                 gender: str,
                 income: int,
                 campaign_channel: str,
                 campaign_type: str,
                 ad_spend: float,
                 click_through_rate: float,
                 conversion_rate: float,
                 website_visits: int,
                 pages_per_visit: float,
                 time_on_site: float,
                 social_shares: int,
                 email_opens: int,
                 email_clicks: int,
                 previous_purchases: int,
                 loyalty_points: int,
                 ):
        self.age = age
        self.gender = gender
        self.income = income
        self.campaign_channel = campaign_channel
        self.campaign_type = campaign_type
        self.ad_spend = ad_spend
        self.click_through_rate = click_through_rate 
        self.conversion_rate = conversion_rate
        self.website_visits = website_visits
        self.pages_per_visit = pages_per_visit
        self.time_on_site = time_on_site
        self.social_shares = social_shares
        self.email_opens = email_opens
        self.email_clicks = email_clicks
        self.previous_purchases = previous_purchases
        self.loyalty_points = loyalty_points

    def get_data_as_data_frame(self):
        try:
            data_dict={
                'Age': [self.age],
                'Gender': [self.gender],
                'Income': [self.income],
                'CampaignChannel': [self.campaign_channel],
                'CampaignType': [self.campaign_type],
                'AdSpend': [self.ad_spend],
                'ClickThroughRate': [self.click_through_rate],
                'ConversionRate': [self.conversion_rate],
                'WebsiteVisits': [self.website_visits],
                'PagesPerVisit': [self.pages_per_visit],
                'TimeOnSite': [self.time_on_site],
                'SocialShares': [self.social_shares],
                'EmailOpens': [self.email_opens],
                'EmailClicks': [self.email_clicks],
                'PreviousPurchases': [self.previous_purchases],
                'LoyaltyPoints': [self.loyalty_points]
            }

            return pd.DataFrame(data_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
