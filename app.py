import joblib
import re
import nltk
import cv2
import time
import math
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import mediapipe as mp
import numpy as np
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from ultralytics import YOLO

# SETUP
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

st.set_page_config(page_title="Multi Project App", layout="wide")
st.title("Multi-Project Streamlit App")

#TABS
tabs = st.tabs(["Sentiment Analysis", "Finger Counter", "Trash Detection"])

#1. SENTIMENT ANALYSIS
with tabs[0]:
    st.header("Sentiment Analysis")
    st.write("This app uses a Support Vector Machine (SVM) model to classify the sentiment of your comments as positive, neutral, or negative.")
    df = pd.read_csv('cleaned_sentiment_dataset.csv')
    model = joblib.load('svm_model.pkl')
    vector = joblib.load('tfidf_vectorizer.pkl')

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)

    text_input = st.text_area("Enter your comment here:", height=100)
    if text_input:
        cleaned_text = preprocess_text(text_input)
        vectorized_text = vector.transform([cleaned_text])
        prediction = model.predict(vectorized_text)
        sentiment = prediction[0]
        
        if sentiment == "positive":
            color = 'green'
        elif sentiment == "neutral":
            color = 'yellow'
        else:
            color = 'red'
            
        st.markdown(
            f"#### Sentiment: <span style='color:{color}; font-weight:bold;'>{sentiment.capitalize()}</span>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("#### Sentiment: -")

    st.subheader("Dataset Insights")
    col1, col2 = st.columns(2)
    with col1:
        fig_platform = px.pie(df, names='Platform', title='Platform Distribution')
        st.plotly_chart(fig_platform, use_container_width=True)
    with col2:
        fig_sentiment = px.pie(df, names='sentiment', title='Sentiment Distribution')
        st.plotly_chart(fig_sentiment, use_container_width=True)

    platform_choice = st.selectbox("Filter by Platform:", df['Platform'].unique())
    filtered_df = df[df['Platform'] == platform_choice]
    st.subheader(f"Sentiment for {platform_choice}")
    st.plotly_chart(px.pie(filtered_df, names='sentiment'), use_container_width=True)

# 2. FINGER COUNTER
with tabs[1]:
    st.header("Real-Time Finger Counter (Webcam)")
    run_finger_counter = st.button("Start Finger Counter")

    if run_finger_counter:
        cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        FINGER_TIPS = [8, 12, 16, 20]
        FINGER_JOINTS = [6, 10, 14, 18]
        THUMB_TIP_ID = 4
        THUMB_JOINT_ID = 3
        FIST_DISTANCE_THRESHOLD = 0.09
        prev_time = 0

        def distance(p1, p2):
            return math.hypot(p1.x - p2.x, p1.y - p2.y)

        def is_thumb_extended(hand_landmarks):
            cmc = hand_landmarks.landmark[1]
            mcp = hand_landmarks.landmark[2]
            tip = hand_landmarks.landmark[4]

            # Vectors
            vec1 = [mcp.x - cmc.x, mcp.y - cmc.y]
            vec2 = [tip.x - mcp.x, tip.y - mcp.y]

            # Angle between vectors
            dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]
            mag1 = math.hypot(*vec1)
            mag2 = math.hypot(*vec2)
            angle = math.degrees(math.acos(dot / (mag1 * mag2 + 1e-6)))
            return angle < 50  # Thumb extended

        #fitsbump
        def is_fist(hand_landmarks, wrist):
            for tip_id in FINGER_TIPS + [THUMB_TIP_ID]:
                if distance(hand_landmarks.landmark[tip_id], wrist) > FIST_DISTANCE_THRESHOLD:
                    return False
            return True

        def count_fingers(hand_landmarks):
            wrist = hand_landmarks.landmark[0]
            count = 0

            # Count fingers
            for tip_id, joint_id in zip(FINGER_TIPS, FINGER_JOINTS):
                tip = hand_landmarks.landmark[tip_id]
                joint = hand_landmarks.landmark[joint_id]
                if tip.y < joint.y and distance(tip, wrist) > 0.1:
                    count += 1

            # Thumb
            if is_thumb_extended(hand_landmarks):
                count += 1

            # Fistbump check
            if is_fist(hand_landmarks, wrist):
                return 0

            return count

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            total_fingers = 0

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    total_fingers += count_fingers(hand_landmarks)

                    # line and dot in hand
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), # Dot
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2) # Line
                    )

            # finger count
            cv2.putText(frame, f'Fingers: {total_fingers}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            cv2.putText(frame, f'FPS: {int(fps)}', (frame.shape[1] - 100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
# 3. TRASH DETECTION
model = YOLO("best.pt")

def resize_image(image, max_width=500):
    width_percent = max_width / float(image.width)
    height = int((float(image.height) * float(width_percent)))
    return image.resize((max_width, height))

def detect_objects(img):
    results = model.predict(source=img, save=False, conf=0.25)[0]
    return results.plot()

# Streamlit layout
with tabs[2]:
    st.markdown("<h2 style='font-size:28px;'>Trash Detection</h2>", unsafe_allow_html=True)
    st.write("This app uses a YOLOv8 model to detect trash in images or real-time webcam feed. it divides the trash into 3 categories: B3, Organic and Anorganik (Inorganic)")
    menu = st.selectbox("Choose Mode:", ["Upload Image", "Webcam Real-time"])

    if menu == "Upload Image":
        st.subheader("Upload or Choose Sample Image for Detection")
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Select Image Source**")
            sample_image_dir = "sample_images"
            sample_image_options = os.listdir(sample_image_dir)
            sample_choice = st.radio("Select Image Source:", ["Upload Your Own", "Use Sample Image"])

            uploaded_file = None
            image = None

            if sample_choice == "Upload Your Own":
                uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert("RGB")
                    resized_input_image = resize_image(image)
                    st.image(resized_input_image, caption="Uploaded Image", use_container_width=False)

            else:  # Sample image
                selected_sample = st.selectbox("Choose a sample image:", sample_image_options)
                image_path = os.path.join(sample_image_dir, selected_sample)
                image = Image.open(image_path).convert("RGB")
                resized_input_image = resize_image(image)
                st.image(resized_input_image, caption=f"Sample Image: {selected_sample}", use_container_width=False)

        with col2:
            st.markdown("**Detection Result**")
            if image is not None:
                st.markdown("Click 'Detect Objects' below to process the image.")
            else:
                st.warning("Please select or upload an image first.")

        st.markdown("---")
        detect_btn = st.button("Detect Objects")

        if detect_btn and image is not None:
            result_image = detect_objects(np.array(image))
            resized_result_image = resize_image(Image.fromarray(result_image))
            with col2:
                st.image(resized_result_image, caption="Detection Result", use_container_width=False)

    elif menu == "Webcam Real-time":
        run_webcam = st.button("Start Webcam")
        if run_webcam:
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result_frame = detect_objects(frame)
                cv2.imshow("Trash Detection", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()