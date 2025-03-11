import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, r2_score
from sklearn.neighbors import KNeighborsRegressor
import joblib

# Load datasets
train_data = pd.read_csv('/content/drive/MyDrive/OOP/train.csv')
val_data = pd.read_csv('/content/drive/MyDrive/OOP/val.csv')
test_data = pd.read_csv('/content/drive/MyDrive/OOP/test.csv')

# Data cleaning function
def clean_data(df):
    df = df.dropna(subset=['User_ID', 'Dominant_Emotion'])
    return df.dropna()

# Clean data
train_data = clean_data(train_data)
val_data = clean_data(val_data)
test_data = clean_data(test_data)

# Label encoding for Dominant_Emotion
all_labels = pd.concat([train_data['Dominant_Emotion'], val_data['Dominant_Emotion'], test_data['Dominant_Emotion']])
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
train_data['Dominant_Emotion'] = label_encoder.transform(train_data['Dominant_Emotion'])
val_data['Dominant_Emotion'] = label_encoder.transform(val_data['Dominant_Emotion'])
test_data['Dominant_Emotion'] = label_encoder.transform(test_data['Dominant_Emotion'])

# Define feature columns
features = ['Age', 'Gender', 'Platform', 'Daily_Usage_Time (minutes)', 'Posts_Per_Day',
            'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']

X_train = train_data[features]
y_train = train_data['Dominant_Emotion']
X_val = val_data[features]
y_val = val_data['Dominant_Emotion']
X_test = test_data[features]
y_test = test_data['Dominant_Emotion']

# Preprocessing
categorical_features = ['Gender', 'Platform']
numerical_features = ['Age', 'Daily_Usage_Time (minutes)', 'Posts_Per_Day',
                      'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

# Train KNN model
knn_model = KNeighborsRegressor(n_neighbors=5, metric='minkowski', p=2)
knn_model.fit(X_train, y_train)

# Predictions
y_val_pred = knn_model.predict(X_val)
y_test_pred = knn_model.predict(X_test)

# Metrics calculation
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, mae, r2, rmse

val_metrics = regression_metrics(y_val, y_val_pred)
test_metrics = regression_metrics(y_test, y_test_pred)

print(f'Validation Metrics (MSE, MAE, R², RMSE): {val_metrics}')
print(f'Test Metrics (MSE, MAE, R², RMSE): {test_metrics}')

# Classification accuracy
def classification_accuracy(y_true, y_pred, label_encoder):
    y_pred_rounded = np.clip(np.round(y_pred).astype(int), 0, len(label_encoder.classes_) - 1)
    y_pred_labels = label_encoder.inverse_transform(y_pred_rounded)
    y_true_labels = label_encoder.inverse_transform(y_true)
    return accuracy_score(y_true_labels, y_pred_labels)

val_accuracy = classification_accuracy(y_val, y_val_pred, label_encoder)
test_accuracy = classification_accuracy(y_test, y_test_pred, label_encoder)

print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Save the model and label encoder
joblib.dump(knn_model, '/content/drive/MyDrive/OOP/knn_model.pkl')
joblib.dump(label_encoder, '/content/drive/MyDrive/OOP/label_encoder_KNN.pkl')

# Download the model and label encoder
from google.colab import files
files.download('/content/drive/MyDrive/OOP/knn_model.pkl')
files.download('/content/drive/MyDrive/OOP/label_encoder_KNN.pkl')
