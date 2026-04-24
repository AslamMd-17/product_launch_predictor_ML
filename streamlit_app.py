import streamlit as st
import joblib
import numpy as np
import os

# Page config
st.set_page_config(
    page_title="Product Launch Predictor",
    layout="centered"
)

# Load model
model = joblib.load(
    os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
)

# Encoding maps — same as training
COMPETITION_MAP = {'Low': 2, 'Medium': 1, 'High': 0}
CATEGORY_MAP    = {'Electronics': 0, 'Clothing': 1, 'Food': 2, 'Health': 3}
TIMING_MAP      = {'Festival': 1, 'Normal': 0}
REGION_MAP      = {'Urban': 1, 'Rural': 0}

# Header
st.title("Product Launch Success Predictor")
st.markdown("Predicted By Random Forest  with (91.25% accuracy)")
st.divider()

# Input form
st.subheader("Enter Product Details")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input(
        "Price (₹)", min_value=500, max_value=50000,
        value=8000, step=500
    )
    competition = st.selectbox(
        "Competition Level", ["Low", "Medium", "High"]
    )
    timing = st.selectbox(
        "Launch Timing", ["Festival", "Normal"]
    )

with col2:
    budget = st.number_input(
        "Marketing Budget (₹)", min_value=1000, max_value=500000,
        value=250000, step=5000
    )
    category = st.selectbox(
        "Product Category", ["Electronics", "Clothing", "Food", "Health"]
    )
    region = st.selectbox(
        "Target Region", ["Urban", "Rural"]
    )

st.divider()

# Predict button
if st.button("Predict Launch Outcome", type="primary", use_container_width=True):
    X = np.array([[
        price,
        budget,
        COMPETITION_MAP[competition],
        CATEGORY_MAP[category],
        TIMING_MAP[timing],
        REGION_MAP[region]
    ]])

    prediction  = model.predict(X)[0]
    probs       = model.predict_proba(X)[0]
    confidence  = probs[prediction] * 100

    st.divider()

    if prediction == 1:
        st.success(f"## !! SUCCESS")
        st.metric("Confidence", f"{confidence:.1f}%")
        st.info("This product launch is predicted to **succeed** based on the given parameters.")
    else:
        st.error(f"## $$ FAILURE")
        st.metric("Confidence", f"{confidence:.1f}%")
        st.warning("This product launch is predicted to **fail**. Consider improving competition conditions or marketing budget.")

# Model info section
st.divider()
with st.expander("About this model"):
    st.markdown("""
    **Algorithm:** Random Forest (100 trees)
    
    | Model | Accuracy | F1 Score |
    |---|---|---|
    | Logistic Regression | 87.50% | 86.84% |
    | Decision Tree | 85.00% | 84.21% |
    | **Random Forest** | **91.25%** | **90.91%** |
    
    **Top predictors:** Competition level and Marketing budget  
    **Dataset:** 400 synthetic records with realistic business logic  
    **Built with:** Python, scikit-learn, Streamlit
    """)