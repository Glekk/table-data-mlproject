# Table Data ML Project

## Dataset Overview

### Features:

### Demographic Information

- **CustomerID**: Unique identifier for each customer.
- **Age**: Age of the customer.
- **Gender**: Gender of the customer (Male/Female).
- **Income**: Annual income of the customer in USD.

### Marketing-specific Variables

- **CampaignChannel**: The channel through which the marketing campaign is delivered (Email, Social Media, SEO, PPC, Referral).
- **CampaignType**: Type of the marketing campaign (Awareness, Consideration, Conversion, Retention).
- **AdSpend**: Amount spent on the marketing campaign in USD.
- **ClickThroughRate**: Rate at which customers click on the marketing content.
- **ConversionRate**: Rate at which clicks convert to desired actions (e.g., purchases).
- **AdvertisingPlatform**: Confidential.
- **AdvertisingTool**: Confidential.

### Customer Engagement Variables

- **WebsiteVisits**: Number of visits to the website.
- **PagesPerVisit**: Average number of pages visited per session.
- **TimeOnSite**: Average time spent on the website per visit (in minutes).
- **SocialShares**: Number of times the marketing content was shared on social media.
- **EmailOpens**: Number of times marketing emails were opened.
- **EmailClicks**: Number of times links in marketing emails were clicked.

### Historical Data

- **PreviousPurchases**: Number of previous purchases made by the customer.
- **LoyaltyPoints**: Number of loyalty points accumulated by the customer.

### Target Variable

- **Conversion**: Binary variable indicating whether the customer converted (1) or not (0).

## Project Overview

### Data Ingestion

- Load the dataset from csv into a pandas DataFrame.
- Split the data into training and testing sets and save them as separate csv files.

### Data Preprocessing

- Handle missing values using SimpleImputer.
- Encode categorical variables using OneHotEncoder.
- Scale numerical features using StandardScaler.
- Save the preprocessing pipeline as a pickle file.

### Model Development

- Train multiple classification models (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier) on the training data.
- Using RandomizedSearchCV, find the best hyperparameters for each model and get the best model out of them.
- Save the best model as a pickle file.

### Prediction

- Load the best model and the preprocessing pipeline from their respective pickle files.
- Make predictions on the test data using the best model.

### App

- Created a Flask app with a simple UI to take input from the user and show the prediction result.



