# YouTube Ad Review Prediction

## Overview
This project implements a **Machine Learning model** to predict the sentiment (positive/negative) of YouTube ad reviews. The model uses **Logistic Regression** with **TF-IDF vectorization** for feature extraction.

## Features
- **Text Preprocessing**: Tokenization, stopword removal, and punctuation cleaning.
- **TF-IDF Vectorization**: Converts text into numerical form.
- **Machine Learning Model**: Logistic Regression trained on labeled review data.
- **Performance Metrics**: Accuracy, Precision, Recall, and F1-Score.
- **Model Persistence**: Save & load trained models using `joblib`.

## Installation
```bash
# Clone the repository
git clone https://github.com/Skilllz/Youtube-adview-prediction.git
cd youtube-ad-review-prediction

# Install required dependencies
pip install -r requirements.txt
```

## Usage
### Train the Model
```bash
python train_model.py
```

### Make Predictions
```python
import joblib
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('youtube_ad_review_model.pkl')

sample_review = ["This ad was amazing! I loved it."]
processed_review = vectorizer.transform(sample_review)
prediction = model.predict(processed_review)
print("Sentiment:", "Positive" if prediction[0] == 1 else "Negative")
```

## Dataset
The dataset should be in CSV format with two columns:
- **review** (text of the review)
- **sentiment** (0 = Negative, 1 = Positive)

## License
This project is licensed under the MIT License.

