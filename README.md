# Multi-Project Streamlit App: Sentiment Analysis, Finger Counter & Trash Detection

An integrated machine learning and computer vision application built with **Streamlit**, combining three distinct features:

- **Sentiment Analysis**  
- **Finger Counter using MediaPipe**  
- **Trash Detection using YOLOv8**

---

## Features

### Sentiment Analysis
- Predicts whether a sentence is **Positive**, **Neutral**, or **Negative**.
- Trained with Logistic Regression, Naive Bayes, and SVM models.
- Displays prediction probabilities for comparison.

### Finger Counter
- Real-time hand tracking using **MediaPipe** and **OpenCV**.
- Detects number of fingers shown from webcam feed.
- Handles both front and back hand, and various hand poses (e.g. peace, fist).

### Trash Detection
- Detects waste categories: **Organic**, **Inorganic**, and **Hazardous (B3)**.
- Works with both **uploaded images** and **webcam (real-time)**.
- Based on custom-trained **YOLOv8** model.

---

## Tech Stack

- **Python**
- **Streamlit** – Web interface
- **scikit-learn** – Sentiment model
- **MediaPipe & OpenCV** – Hand detection
- **Ultralytics YOLOv8** – Object detection
- **Roboflow** – Dataset management and model export
- **Pillow / NumPy** – Image processing

---

## How to Run

1. **Clone this repo**

```
git clone https://github.com/your-username/streamlit-ml-app.git
cd streamlit-ml-app
```
2. **Install Dependencies**
```
pip install -r requirements.txt
``` 
3. **Run the App**
```
streamlit run app.py
```
