import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
import streamlit as st
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('sms-spam-collection.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Load the LSTM model
lstm_model = load_model('best_lstm_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the Logistic Regression model
with open('tuned_lr_model.pkl', 'rb') as handle:
    logistic_model = pickle.load(handle)

# Function to preprocess input text for LSTM
def preprocess_for_lstm(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    return padded

# Function to preprocess input text for Logistic Regression
def preprocess_for_lr(text):
    return vectorizer.transform([text])

# Calculate and display model performance metrics
def display_performance_metrics():
    y_pred_lr = logistic_model.predict(X_test_tfidf)
    lr_report = classification_report(y_test, y_pred_lr, output_dict=True)
    
    st.subheader("Model Performance Metrics")
    st.write("Logistic Regression Accuracy: {:.2f}%".format(lr_report['accuracy'] * 100))
    st.write("Precision: {:.2f}".format(lr_report['1']['precision']))
    st.write("Recall: {:.2f}".format(lr_report['1']['recall']))
    st.write("F1 Score: {:.2f}".format(lr_report['1']['f1-score']))

# Streamlit app layout
st.title("Text Classification App")

st.sidebar.header("User Input")
user_input = st.sidebar.text_area("Enter your text here (separate multiple messages with new lines):")

if st.sidebar.button("Classify"):
    if user_input:
        messages = user_input.split('\n')
        lstm_predictions = []
        lr_predictions = []
        lstm_confidences = []

        for message in messages:
            if message.strip():  # Ignore empty lines
                # LSTM Prediction
                lstm_input = preprocess_for_lstm(message)
                lstm_prob = lstm_model.predict(lstm_input)[0][0]
                lstm_prediction = int(lstm_prob > 0.5)
                lstm_predictions.append("Positive" if lstm_prediction == 1 else "Negative")
                lstm_confidences.append(lstm_prob)  # Store confidence score

                # Logistic Regression Prediction
                lr_input = preprocess_for_lr(message)
                lr_prediction = logistic_model.predict(lr_input)
                lr_predictions.append("Positive" if lr_prediction[0] == 1 else "Negative")

        # Display results
        st.subheader("Results")
        for i, message in enumerate(messages):
            st.write(f"Message: {message.strip()}")
            st.write(f"LSTM Prediction: {lstm_predictions[i]} (Confidence: {lstm_confidences[i]:.2f})")
            st.write(f"Logistic Regression Prediction: {lr_predictions[i]}")

    else:
        st.warning("Please enter some text to classify.")

# Display model performance metrics
display_performance_metrics()
