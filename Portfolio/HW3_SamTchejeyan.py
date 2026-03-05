#!/usr/bin/env python
# coding: utf-8

# # Bitcoin Signals Prediction
# 

# ## Instructions

# * Formulate a strategy to predict Bitcoin Buy, Hold, and Sell signals. State the features that define your strategy (5 pts)
#     * Only use technical indicators, such as trends or momentum.
# * Download the data and add these features to your dataset -- at least 4 features -- your features cannot be identical to the class activity (7.5 pts)
# * Add a few visualizations of the relationships between features (e.g., heatmap of the correlation, scatter matrix, etc.), and the distribution of the data (e.g., histograms, density plots, etc.) -- at least 5 plots (12.5 pts
# * In addition to Logistic Regression, implement different approaches (at least two different approaches) to resolve overfitting due to noise and outliers (20 pts) -- address class imbalance if present
#     * Feature selection
#     * Regularization (e.g., Lasso, Ridge, Elastic Net)
#     * Feature scale transformation (e.g., standardization, mean-centering, log transformation, etc.)
# * Compare the models using a metric (e.g., accuracy, auc, etc.) of your choise as scoring. (10 pts) -- do not forget to adjust the threshold if needed
# * Perform model tuning on logistic regression with any regularization (e.g., Lasso, Ridge, Elastic Net) (10 pts) -- at least two parameters to vary
# * Save the model (5 pts)
# * Rank the features by importance (5 pts)
# * Save the Sharp Explainer and create a sharp plot (5 pts)
# * Deploy the model on AWS (10 pts)
# * Deploy the Streamlit app (10 pts)

# # Bitcoin Signals Prediction
# ## Strategy for Predicting Buy, Hold, Sell
# This model predicts Bitcoin trading signals (Buy, Hold, or Sell) using technical indicators based on historical price data.
# 
# The signal is defined using the next-day return:
# * Buy: next-day return above a positive threshold
# * Sell: next-day return below a negative threshold
# * Hold: return within the threshold range
# 
# The model uses technical indicators as features, including exponential moving averages (trend), rate of change (momentum), rolling standard deviation (volatility), and price relative to a moving average.
# 
# These indicators help capture patterns in Bitcoin price behavior that may indicate future price movement.

# ## Loading the data and python packages

# In[1]:


#Disable the warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:





# In[3]:


# Load libraries
import numpy as np
import pandas as pd

import kagglehub

import shap

from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN

import joblib
import tarfile

import requests, zipfile, io

import os
import sys
import shutil

import boto3
import sagemaker
from sagemaker.image_uris import retrieve
from sklearn.model_selection import train_test_split

from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from sagemaker.sklearn.model import SKLearnModel

import importlib


# In[4]:


import sys
import importlib

# Build the path to the 'src' directory (relative to the notebook's location)
module_path = os.path.abspath('..') 

# Add the 'src' directory to the system path list
if module_path not in sys.path:
    sys.path.append(module_path)


# In[5]:


import src.Custom_Classes
import src.feature_utils
importlib.reload(src.Custom_Classes)
importlib.reload(src.feature_utils)
from src.Custom_Classes import FeatureEngineer
from src.feature_utils import get_bitcoin_historical_prices


# ## Loading the Data

# In[6]:


# load dataset
dataset = pd.read_csv(r'./BitstampData.csv')


# In[7]:


dataset.dropna(subset=['Close'], how='all', inplace=True)
dataset = dataset.reset_index(drop=True)


# In[8]:


dataset=dataset[['Close']]


# ## Data Preparation

# In[9]:


# Calculate next day's simple return
dataset['Next_Day_Return'] = dataset['Close'].pct_change().shift(-1)

# Define a small noise threshold (e.g., 0.1%)
threshold = 0.001

# Create the 3-class target variable
conditions = [
    (dataset['Next_Day_Return'] > threshold),
    (dataset['Next_Day_Return'] < -threshold)
]
choices = [1, -1]  # 1 for Buy, -1 for Sell

# Default value is 0 (Hold/Neutral)
dataset['signal'] = np.select(conditions, choices, default=0)

# Final step: Drop the last row which will have NaN for Next_Day_Return
dataset.dropna(inplace=True)


# In[10]:


# Technical indicator features (Bullet 2)

# Trend: EMA
dataset['EMA_10'] = dataset['Close'].ewm(span=10, adjust=False).mean()

# Momentum: 5-day rate of change
dataset['ROC_5'] = dataset['Close'].pct_change(5)

# Volatility: 10-day rolling std of daily returns
dataset['Volatility_10'] = dataset['Close'].pct_change().rolling(10).std()

# Price positioning: price relative to 10-day moving average
dataset['Price_vs_MA10'] = dataset['Close'] / dataset['Close'].rolling(10).mean()

# Drop rows with NaNs created by rolling calculations
dataset.dropna(inplace=True)


# In[11]:


features = FeatureEngineer(windows=[10]).transform(dataset[['Close']])
features.head()


# ## Data Visualization

# ***Update this part accordingly***

# In[12]:


dataset[['Close']].plot(grid=True)
plt.show()


# In[13]:


fig = plt.figure()
plot = dataset.groupby(['signal']).size().plot(kind='barh', color='red')
plt.show()


# In[14]:


# Correlation heatmap between features
plt.figure(figsize=(8,6))
sns.heatmap(
    dataset[['EMA_10', 'ROC_5', 'Volatility_10', 'Price_vs_MA10']].corr(),
    annot=True, cmap='coolwarm'
)
plt.title("Feature Correlation Heatmap")
plt.show()


# In[15]:


# Histograms of feature distributions
dataset[['EMA_10', 'ROC_5', 'Volatility_10', 'Price_vs_MA10']].hist(figsize=(10,8))
plt.suptitle("Feature Distributions")
plt.show()


# In[16]:


# Scatter matrix for feature relationships
scatter_matrix(
    dataset[['EMA_10', 'ROC_5', 'Volatility_10', 'Price_vs_MA10']],
    figsize=(10,8)
)
plt.show()


# ## Evaluate Algorithms and Models

# ### Train Test Split and Evaluation Metrics

# ***Update this part accordingly***

# In[17]:


# split out validation dataset for the end
subset_dataset= dataset.iloc[-1000:]
Y = subset_dataset["signal"]
X = subset_dataset[['EMA_10', 'ROC_5', 'Volatility_10', 'Price_vs_MA10']]
validation_size = 0.2
train_size = int(len(X) * (1-validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]


# ### Define Pipeline

# In[18]:


pipeline_steps_baseline = [    
    # Handle missing values
    ('imputer', SimpleImputer(strategy='mean')),
        
    # Transform the scale (Will now scale the MA/MOM features as well)
    ('scaler', StandardScaler()),
    
    # Resampling (Addressing Imbalance - MUST come before the model)
    ('sampler', SMOTE(random_state=42, sampling_strategy='minority')),
    
    # Estimator
    ('model', LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=5000))
]


# In[19]:


pipeline_steps_feat_selection = [    
    # Handle missing values
    ('imputer', SimpleImputer(strategy='mean')),
    
    # Transform the scale (Will now scale the MA/MOM features as well)
    ('scaler', StandardScaler()),
    
    # Feature Selection
    ('feature_selection', SelectKBest(score_func=mutual_info_classif, k=3)),
    
    # Resampling (Addressing Imbalance - MUST come before the model)
    ('sampler', SMOTE(random_state=42, sampling_strategy='minority')),
    
    # Estimator
    ('model', LogisticRegression(penalty='l1', solver='saga', C=1.0, class_weight='balanced', max_iter=5000))
]


# In[20]:


pipeline_steps = [('BASELINE',pipeline_steps_baseline),
                  ('FEAT_SELECTION', pipeline_steps_feat_selection)]


# ### Compare Models and Algorithms

# In[21]:


# test options for classification
num_folds = 10
seed = 7
scoring = 'f1_weighted'
#scoring = 'f1_weighted'
#scoring = 'precision'
#scoring = 'recall'
#scoring ='neg_log_loss'
#scoring = 'roc_auc'


# In[22]:


num_folds = 10
seed = 7

names = []
kfold_results = []
test_results = []
train_results = []
for name, item  in pipeline_steps:
    names.append(name)
    stock_pipeline = Pipeline(item)

    kfold = KFold(n_splits=num_folds, shuffle=False)
    cv_results = -1*cross_val_score(estimator=stock_pipeline,X=X_train,y=Y_train,scoring=scoring,cv=kfold)
    kfold_results.append(cv_results)

    stock_pipeline.fit(X_train, Y_train)
    y_train_pred = stock_pipeline.predict(X_train)
    y_test_pred = stock_pipeline.predict(X_test)
    
    train_result = f1_score(Y_train, y_train_pred, average='weighted')
    test_result  = f1_score(Y_test,  y_test_pred,  average='weighted')

    train_results.append(train_result)
    test_results.append(test_result)
    
    msg = "%s: %f (%f) %f %f" % (name, cv_results.mean(), cv_results.std(), train_result, test_result)
    print(msg)


# In[23]:


pd.DataFrame(np.array(kfold_results).T, columns=names).plot(kind='box')
plt.show()


# In[24]:


results = pd.DataFrame(columns=names)
results.loc['Train F1 Weighted'] = train_results
results.loc['Test F1 Weighted'] = test_results
results.plot(kind='bar')
plt.show()


# ## Model Tuning and Grid Search

# ***Adjust the code below accordingly***

# In[25]:


pipeline_steps_feat_selection = [
    # Handle missing values
    ('imputer', SimpleImputer(strategy='mean')),

    # Scale features
    ('scaler', StandardScaler()),

    # Feature Selection
    ('feature_selection', SelectKBest(score_func=mutual_info_classif)),

    # Address class imbalance
    ('sampler', SMOTE(random_state=42)),

    # Logistic Regression with regularization
    ('model', LogisticRegression(solver='saga', max_iter=5000))
]

classification_pipeline = Pipeline(pipeline_steps_feat_selection)


# In[26]:


param_grid = [
    {
        # Tuning SelectKBest: How many features should we keep?
        'feature_selection__k': [1, 2, 3, 5, 'all'],

        # Logistic Regression regularization strength
        'model__C': [0.01, 0.1, 1, 10],

        # Regularization type
        'model__penalty': ['l1', 'l2']
    }
]


# In[27]:


num_folds = 10
scoring = 'f1_weighted'

kfold = KFold(n_splits=num_folds, shuffle=False)

grid_search = GridSearchCV(
    estimator=classification_pipeline,
    param_grid=param_grid,
    cv=kfold,
    scoring=scoring
)
grid_search.fit(X_train,Y_train)


# In[28]:


print("\n--- Best Results ---")
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))


# ## Save the Best Model

# In[29]:


# Save Model Using JobLib
from joblib import dump
from joblib import load

best_pipeline = grid_search.best_estimator_

filename = r'./finalized_bitcoin_model.joblib'
dump(best_pipeline, filename)


# In[30]:


import tarfile

with tarfile.open('finalized_bitcoin_model.tar.gz', 'w:gz') as tar:
    
    # Add the serialized model file
    tar.add(r'./finalized_bitcoin_model.joblib', arcname='finalized_bitcoin_model.joblib')
    
    # Add the custom source code directory
    tar.add(r'../src', arcname='src')


# ## Variable Importance

# In[31]:


feature_names = best_pipeline[1:4].get_feature_names_out()
Importance = pd.DataFrame({'Importance':np.abs(best_pipeline.named_steps['model'].coef_[1])*100}, index=feature_names)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r' )
plt.xlabel('Variable Importance')


# ## SHAP Plot

# In[32]:


model = best_pipeline.named_steps['model']

# We run only the preprocessing steps
preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
X_train_transformed = preprocessing_pipeline.transform(X_train)

explainer = shap.Explainer(model, X_train_transformed)

with open("./explainer_bitcoin.shap", "wb") as f:
    explainer.save(f)


# In[33]:


X_test_transformed = preprocessing_pipeline.transform(X_test)
feature_names = best_pipeline[1:4].get_feature_names_out()
X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
shap_values = explainer(X_test_transformed)

# Now this simplified line will work and include names
shap.plots.waterfall(shap_values[199, :, 1])


# In[34]:


shap_values.shape


# ## Model Deployment on AWS

# In[35]:


session = boto3.Session()
s3_client = session.client('s3')
bucket_name='sam-tchejeyan-s3-bucket'
sagemaker_session=sagemaker.Session(boto_session=session, default_bucket=bucket_name)

credentials = session.get_credentials()

current_access_key = credentials.access_key
current_secret_key = credentials.secret_key
current_session_token = credentials.get_frozen_credentials().token

print(f"Your Current Access Key ID is: {current_access_key}")
print(f"Your Current Secret Key ID is: {current_secret_key}")
print(f"Your Session Token is:\n{current_session_token}")


# In[36]:


s3_resource = boto3.resource('s3')
s3_bucket = s3_resource.Bucket(bucket_name)
s3_bucket.objects.all().delete()


# In[37]:


# save explainer
s3_client.upload_file(
    Filename="./explainer_bitcoin.shap",
    Bucket=bucket_name,
    Key = "explainer/explainer_bitcoin.shap")


# In[38]:


s3_path_key = r'sklearn-pipeline-deployment'
filename = r'finalized_bitcoin_model.tar.gz'
s3_client.upload_file(
        Filename=filename, 
        Bucket=bucket_name, 
        Key= f"{s3_path_key}/{os.path.basename(filename)}")


# In[39]:


model_s3_uri = f"s3://{bucket_name}/{s3_path_key}/{filename}"
model_s3_uri


# In[40]:


with open('requirements.txt', 'w') as f:
    f.write('numpy==1.26.4\n')
    f.write('scipy==1.12.0\n')
    f.write('scikit-learn==1.3.2\n')
    f.write('statsmodels==0.14.1\n')
    f.write('pandas==2.2.0\n')
    f.write('imbalanced-learn==0.12.0\n') # This provides 'imblearn'


# In[41]:


model_name = 'HW3-Model'
endpoint_name = 'HW3-pipeline-endpoint-auto'
instance_type = 'ml.m5.large' # Use a cost-effective instance for hosting
framework_version = '1.2-1' 
custom_code_uri = "s3://{}/customCode/".format(bucket_name)

sklearn_model = SKLearnModel(
    model_data=model_s3_uri, # <-- Using the custom URI here
    role=sagemaker.get_execution_role(),
    entry_point='inference_bitcoin.py', 
    framework_version='1.2-1', 
    py_version="py3",
    dependencies=["requirements.txt"],
    source_dir='.',
    name=model_name,
    sagemaker_session=sagemaker_session,
    code_location=custom_code_uri
)


# In[42]:


print(f"\n4. Starting deployment of model {model_name} to endpoint {endpoint_name}...")
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    endpoint_name=endpoint_name,
    #serializer=CSVSerializer(), # Expecting CSV input for prediction
    #deserializer=JSONDeserializer() # Assuming your model outputs JSON
)
print("\nDeployment complete! You can now invoke the endpoint:")
print(f"Endpoint Name: {endpoint_name}")


# ## Deploy the Streamlit App

# In[ ]:





# In[ ]:




