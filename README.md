# 🚀 Product Launch Success Predictor

A machine learning system that predicts whether a product launch will succeed or fail based on 6 business parameters.

## Live Demo
👉 [Click here to try the app](https://success-launch-predictor.streamlit.app/)

## Model Performance
| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 87.50% | 86.84% |
| Decision Tree | 85.00% | 84.21% |
| **Random Forest** | **91.25%** | **90.91%** |

## Features
- Trained 3 ML models and compared performance
- Demonstrated overfitting with Decision Tree (15% train-test gap reduced to 9.7% after pruning)
- Feature importance analysis — competition and marketing budget are strongest predictors
- Deployed as interactive web application

## Tech Stack
Python · scikit-learn · Random Forest · Streamlit · pandas · numpy · matplotlib · seaborn

## Run Locally
pip install -r requirements.txt
streamlit run streamlit_app.py