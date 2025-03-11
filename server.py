from flask import Flask, request, jsonify
import pandas as pd
import joblib  # or use pickle if the model was saved using pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = 'KNN\knn_model.pkl'  # Adjust the path if needed
model = joblib.load(model_path)

# Define label encoder to decode predicted numerical labels back to original emotions
label_encoder = joblib.load('KNN\label_encoder_KNN.pkl')  # Save and load LabelEncoder if it's separate

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input data from request
    try:
        data = request.form  # Expecting URL-encoded data
        age = int(data.get('age'))
        gender = data.get('gender')
        platform = data.get('platform')
        daily_usage = int(data.get('daily_usage'))
        posts = int(data.get('posts'))
        likes = int(data.get('likes'))
        comments = int(data.get('comments'))
        messages = int(data.get('messages'))
        
        # Create a DataFrame from the input data
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Platform': platform,
            'Daily_Usage_Time (minutes)': daily_usage,
            'Posts_Per_Day': posts,
            'Likes_Received_Per_Day': likes,
            'Comments_Received_Per_Day': comments,
            'Messages_Sent_Per_Day': messages
        }])

        # Make prediction
        prediction = model.predict(input_data)

        # Convert numerical prediction to emotion label
        predicted_emotion = label_encoder.inverse_transform([int(prediction[0])])[0]

        return predicted_emotion

    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
