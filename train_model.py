# train_model.py
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("cleaned_data.csv")  # Use your actual CSV filename

# Features and target
X = df[['location', 'total_sqft', 'bath', 'BHK']]
y = df['price']

# Preprocessing and model pipeline
column_transformer = ColumnTransformer(
    transformers=[('location', OneHotEncoder(handle_unknown='ignore'), ['location'])],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', column_transformer),
    ('model', LinearRegression())
])

# Train model
pipeline.fit(X, y)

# Save the model
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
