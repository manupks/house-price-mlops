import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('house_data.csv')

# Drop outliers
df = df[(df['total_sqft'] > 300) & (df['price'] > 10)]

# Replace rare locations with 'other'
location_counts = df['location'].value_counts()
rare_locations = location_counts[location_counts <= 10].index
df['location'] = df['location'].apply(lambda x: 'other' if x in rare_locations else x)

# Features and target
X = df[['location', 'total_sqft', 'bath', 'BHK']]
y = df['price']

# Save list of known locations for Streamlit
locations = sorted(df['location'].unique())
with open("locations.pkl", "wb") as f:
    pickle.dump(locations, f)

# Column transformer
column_transformer = ColumnTransformer([
    ('location_ohe', OneHotEncoder(handle_unknown='ignore'), ['location'])
], remainder='passthrough')

# Pipeline with Random Forest
pipe = Pipeline(steps=[
    ('preprocessor', column_transformer),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")

# Save model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(pipe, f)
