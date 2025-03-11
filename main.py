import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

# Load data
train_data = pd.read_csv('/content/drive/MyDrive/OOP/train.csv')
val_data = pd.read_csv('/content/drive/MyDrive/OOP/val.csv')
test_data = pd.read_csv('/content/drive/MyDrive/OOP/test.csv')

# Data Cleaning
def clean_data(df):
    # Drop rows where User_ID or Dominant_Emotion is missing
    df = df.dropna(subset=['User_ID', 'Dominant_Emotion'])

    # Drop any additional non-feature columns if present
    df = df.drop(columns=[col for col in df.columns if 'Rows' in col or 'Columns' in col or col.isdigit()], errors='ignore')

    # Drop rows with missing values in essential columns
    df = df.dropna()
    return df

# Clean train, val, and test sets
train_data = clean_data(train_data)
val_data = clean_data(val_data)
test_data = clean_data(test_data)

# Combine labels from all datasets to ensure LabelEncoder sees all possible labels
all_labels = pd.concat([train_data['Dominant_Emotion'], val_data['Dominant_Emotion'], test_data['Dominant_Emotion']])

# Fit LabelEncoder on all unique labels in the target
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Encode Dominant_Emotion in each dataset
train_data['Dominant_Emotion'] = label_encoder.transform(train_data['Dominant_Emotion'])
val_data['Dominant_Emotion'] = label_encoder.transform(val_data['Dominant_Emotion'])
test_data['Dominant_Emotion'] = label_encoder.transform(test_data['Dominant_Emotion'])

# Define the feature columns
features = ['Age', 'Gender', 'Platform', 'Daily_Usage_Time (minutes)', 'Posts_Per_Day',
            'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']

# Separate features (X) and target (y)
X_train = train_data[features]
y_train = train_data['Dominant_Emotion']
X_val = val_data[features]
y_val = val_data['Dominant_Emotion']
X_test = test_data[features]
y_test = test_data['Dominant_Emotion']

# Preprocessing for categorical features
categorical_features = ['Gender', 'Platform']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing for numerical features
numerical_features = ['Age', 'Daily_Usage_Time (minutes)', 'Posts_Per_Day',
                      'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
]) #Imputes NA values 

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(random_state=0))])

# Train the model
model.fit(X_train, y_train)

# Prediction and Evaluation on the validation set
y_val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
print(f'Validation Mean Absolute Error: {val_mae}')

# Testing on the test set
y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
print(f'Test Mean Absolute Error: {test_mae}')

# -------
from sklearn.metrics import mean_absolute_error, accuracy_score

# Step 1: MAE (already calculated previously)
y_val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
print(f'Validation Mean Absolute Error (MAE): {val_mae}')

# Step 2: Round predictions to nearest integer for discrete labels
# Convert predicted regression values to integer labels
y_val_pred_rounded = np.round(y_val_pred).astype(int)
y_val_pred_rounded = np.clip(y_val_pred_rounded, 0, len(label_encoder.classes_) - 1)  # Ensure predictions are within label range

# Convert integer predictions back to original labels
y_val_labels = label_encoder.inverse_transform(y_val)
y_val_pred_labels = label_encoder.inverse_transform(y_val_pred_rounded)

# Calculate accuracy as the proportion of correct predictions
val_accuracy = accuracy_score(y_val_labels, y_val_pred_labels)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Repeat the same for the test set
y_test_pred = model.predict(X_test)
y_test_pred_rounded = np.round(y_test_pred).astype(int)
y_test_pred_rounded = np.clip(y_test_pred_rounded, 0, len(label_encoder.classes_) - 1)

# Convert predictions and actuals back to labels
y_test_labels = label_encoder.inverse_transform(y_test)
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred_rounded)

# Calculate accuracy on the test set
test_accuracy = accuracy_score(y_test_labels, y_test_pred_labels)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# ----
import joblib

# Define the path in Google Drive
model_filename = '/content/drive/MyDrive/OOP/trained_model.joblib'

# Save the model to the specified path
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")
loaded_model = joblib.load('/content/drive/MyDrive/OOP/trained_model.joblib')
from google.colab import files
files.download(model_filename)

label_encoder_path = '/content/drive/MyDrive/OOP/label_encoder.pkl'
joblib.dump(label_encoder, label_encoder_path)
from google.colab import files

# Provide the path where you saved the label encoder
files.download(label_encoder_path)