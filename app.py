"""
app.py
------
Streamlit front-end for the Wellness Tourism Package purchase predictor.

Loads the model committed to the repository (model/best_model.joblib),
collects customer / interaction inputs through widgets, assembles them into
a single-row DataFrame in the exact column order the training pipeline used,
and returns a purchase-likelihood prediction. Deployed on Streamlit Community
Cloud, pointed at this repo's app.py with requirements.txt as the
dependency file.
"""

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "model/best_model.joblib"

FEATURE_ORDER = [
    "Age", "TypeofContact", "CityTier", "DurationOfPitch", "Occupation",
    "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def main():
    st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧘")
    st.title("🧘 Wellness Tourism Package — Purchase Predictor")
    st.write(
        "Enter a customer's profile and sales-pitch interaction details to "
        "predict whether they are likely to purchase the new Wellness "
        "Tourism Package."
    )

    model = load_model()

    st.header("Customer details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City tier", [1, 2, 3])
        occupation = st.selectbox(
            "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
        )
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
        designation = st.selectbox(
            "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )
    with col2:
        monthly_income = st.number_input("Monthly income", min_value=0, value=20000, step=500)
        num_persons = st.number_input("Number of persons visiting", min_value=1, max_value=10, value=2)
        num_children = st.number_input("Number of children (under 5) visiting", min_value=0, max_value=5, value=0)
        num_trips = st.number_input("Average number of trips per year", min_value=0, max_value=20, value=2)
        passport = st.selectbox("Holds a passport?", ["No", "Yes"])
        own_car = st.selectbox("Owns a car?", ["No", "Yes"])
        preferred_star = st.selectbox("Preferred property star rating", [3, 4, 5])

    st.header("Sales pitch interaction")
    col3, col4 = st.columns(2)
    with col3:
        product_pitched = st.selectbox(
            "Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
        )
        duration_of_pitch = st.number_input("Duration of pitch (minutes)", min_value=1, max_value=60, value=15)
    with col4:
        num_followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=3)
        satisfaction_score = st.selectbox("Pitch satisfaction score", [1, 2, 3, 4, 5])

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "Age": age,
            "TypeofContact": type_of_contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": num_persons,
            "NumberOfFollowups": num_followups,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": preferred_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": num_trips,
            "Passport": 1 if passport == "Yes" else 0,
            "PitchSatisfactionScore": satisfaction_score,
            "OwnCar": 1 if own_car == "Yes" else 0,
            "NumberOfChildrenVisiting": num_children,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
        }])[FEATURE_ORDER]

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.success(f"✅ Likely to purchase the Wellness Package (probability: {proba:.1%})")
        else:
            st.warning(f"❌ Unlikely to purchase the Wellness Package (probability: {proba:.1%})")


if __name__ == "__main__":
    main()
