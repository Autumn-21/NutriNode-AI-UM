import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
from catboost import CatBoostRegressor, Pool
from sklearn.neighbors import NearestNeighbors

# Page Configuration
st.set_page_config(
    page_title="NutriNode AI | Personalized Nutrition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Load Models & Schema (Cached) ---
@st.cache_resource
def load_resources():
    # Load Schema
    try:
        with open("schema.json", "r") as f:
            schema = json.load(f)
    except FileNotFoundError:
        st.error("Error: 'schema.json' not found. Please ensure it is in the app directory.")
        return None, None, None

    # Define CatBoost Model Paths
    model_files = {
        "calories": "catboost_reco_calories.cbm",
        "protein": "catboost_reco_protein.cbm",
        "carbs": "catboost_reco_carbs.cbm",
        "fats": "catboost_reco_fats.cbm"
    }
    
    models = {}
    for key, filename in model_files.items():
        if os.path.exists(filename):
            model = CatBoostRegressor()
            model.load_model(filename)
            models[key] = model
        else:
            st.error(f"Error: Model file '{filename}' not found.")
            return None, None, None

    # Load Recipe Data, Scaler AND Gym Data
    rec_resources = {}
    try:
        # Load Recipes
        if os.path.exists("recipes_processed.pkl"):
            rec_resources["recipes"] = joblib.load("recipes_processed.pkl")
        else:
            st.error("Error: 'recipes_processed.pkl' not found.")

        # Load Scaler
        if os.path.exists("wellness_scaler.pkl"):
            rec_resources["scaler"] = joblib.load("wellness_scaler.pkl")
        else:
            st.error("Error: 'wellness_scaler.pkl' not found.")

        # Load Gym Data
        if os.path.exists("GYM_cleaned-1.csv"):
            rec_resources["gym"] = pd.read_csv("GYM_cleaned-1.csv")
        else:
            rec_resources["gym"] = pd.DataFrame() # Empty fallback
                
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None
            
    return schema, models, rec_resources

schema_data, models, rec_resources = load_resources()

# --- Helper Function: Recommend Meals ---
def get_knn_recommendations(daily_cal, daily_prot, daily_fat, daily_carb, rec_resources, diet_habit, goal, n_days=7):
    """
    Filters recipes based on consolidated dietary habits/allergens, 
    then runs KNN to find the closest nutritional matches.
    """
    if not rec_resources or "recipes" not in rec_resources or "scaler" not in rec_resources:
        return None

    df_recipes = rec_resources["recipes"]
    scaler = rec_resources["scaler"]

    # 1. Nutritional Weights based on Gym Goal
    # High weight = KNN prioritizes matching this specific macro more strictly
    weights = np.array([1.0, 1.0, 1.0, 1.0]) # [Calories, Protein, Fat, Carbs]
    if goal == 'Muscle Gain':
        weights = np.array([1.0, 2.5, 0.8, 1.2]) # Prioritize Protein
    elif goal == 'Fat Burn':
        weights = np.array([1.5, 1.2, 1.0, 0.6]) # Prioritize Caloric Deficit & Low Carb

    # 2. Define Meal Targets & Sequence
    # Percentages of total daily intake per meal
    meal_types = [
        ("Breakfast", 0.20, "breakfast"),
        ("Lunch", 0.35, "lunch/dinner"),
        ("Snack", 0.15, "snack"), 
        ("Dinner", 0.30, "lunch/dinner")
    ]
    
    recommendations = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for day_name in days:
        for meal_name, pct, search_term in meal_types:
            
            # Target calories/macros for this specific meal
            target_vector = [
                daily_cal * pct, 
                daily_prot * pct, 
                daily_fat * pct, 
                daily_carb * pct
            ]

            # --- 3. FILTERING LOGIC ---
            # Filter A: Meal Category (Matches 'breakfast', 'snack', etc.)
            mask = df_recipes['meal_type'].apply(lambda x: any(search_term in str(m).lower() for m in x))

            # Filter B: Dietary Habits & Allergens (Vegan, Peanut-Free, etc.)
            # We look for the selection directly inside the 'health_labels' list
            if diet_habit != 'Regular':
                mask &= df_recipes['health_labels'].apply(lambda x: diet_habit in x)

            filtered_df = df_recipes[mask].copy()

            # --- 4. KNN LOGIC ---
            if not filtered_df.empty:
                feature_cols = ['calories_per_person', 'Protein_g', 'Fat_g', 'Carbs_g']
                
                # Scale the filtered recipes and apply the goal weights
                subset_scaled = scaler.transform(filtered_df[feature_cols].fillna(0)) * weights
                target_scaled = scaler.transform([target_vector]) * weights

                # Find the 5 closest nutritional matches
                n_neighbors = min(len(filtered_df), 5)
                knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(subset_scaled)
                distances, indices = knn.kneighbors(target_scaled)

                # Randomly pick one of the top 5 matches to ensure variety across the week
                chosen_idx = np.random.choice(indices[0])
                rec = filtered_df.iloc[chosen_idx]
                
                recommendations.append({
                    "Day": day_name,
                    "Meal": meal_name,
                    "Name": rec['recipe_name'],
                    "Calories": int(rec['calories_per_person']),
                    "Protein": int(rec['Protein_g']),
                    "Carbs": int(rec['Carbs_g']),
                    "Fats": int(rec['Fat_g'])
                })
            else:
                # Fallback if no recipes meet the strict dietary filters
                recommendations.append({
                    "Day": day_name,
                    "Meal": meal_name,
                    "Name": f"No {diet_habit} recipe found for {meal_name}",
                    "Calories": 0, "Protein": 0, "Carbs": 0, "Fats": 0
                })

    return pd.DataFrame(recommendations)

# --- Header & Problem Statement ---
st.title("🤖 NutriNode AI")
st.subheader("Personalized Nutrition & Dietary Intelligence System")
st.markdown("---")

# --- Input Section ---
st.header("User Profile Input")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    gender = st.selectbox("Gender", ["Female", "Male"])
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, step=1)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70, step=1)
    
    # Calculate BMI
    bmi_val = weight / ((height/100)**2)
    
    if bmi_val < 18.5:
        bmi_cat = "UnderWeight"
        bmi_color = "blue"
    elif 18.5 <= bmi_val < 25:
        bmi_cat = "Normal Weight"
        bmi_color = "green"
    elif 25 <= bmi_val < 30:
        bmi_cat = "OverWeight"
        bmi_color = "orange"
    else:
        bmi_cat = "Obesity"
        bmi_color = "red"

    st.markdown(f"**BMI:** {bmi_val:.2f}")
    st.markdown(f"**Category:** :{bmi_color}[{bmi_cat}]")

with col2:
    st.subheader("Health Metrics")
    # Consolidated list including allergens as habits
    dietary_habit = st.selectbox("Dietary Habit", [
        "Regular", 
        "Vegan", 
        "Vegetarian", 
        "Keto",
        "Dairy-Free", 
        "Gluten-Free", 
        "Peanut-Free", 
        "Fish-Free",
        "Egg-Free",
        "Shellfish-Free"
    ])
    
    bp_systolic = st.number_input("Blood Pressure Systolic", min_value=80, max_value=200, value=120, step=1)
    bp_diastolic = st.number_input("Blood Pressure Diastolic", min_value=50, max_value=130, value=80, step=1)
    cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200, step=1)
    blood_sugar = st.number_input("Blood Sugar Level", min_value=50, max_value=300, value=100, step=1)

with col3:
    st.subheader("Lifestyle & Habits")
    gym_goal = st.selectbox("Gym Goal", ["Muscle Gain", "Fat Burn"])
    smoking = st.selectbox("Smoking Habit", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    
    # Updated: Now increases by 500 steps per click/slide
    daily_steps = st.slider("Daily Steps", min_value=0, max_value=20000, value=5000, step=500)
    
    exercise_freq_val = st.slider("Exercise Frequency (days/week)", min_value=0, max_value=7, value=3)
    exercise_freq = str(exercise_freq_val) 
    sleep_hours = st.slider("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

# --- Action ---
st.markdown("---")
generate_btn = st.button("Generate Recommendations", type="primary")

# --- Prediction & Output Section ---
# --- Prediction & Output Section ---
if generate_btn and schema_data and models:
    st.header("Recommendations (AI Predicted)")
    
    # 1. Prepare Input for CatBoost Models
    input_data = {
        "Age": float(age),
        "Gender": gender,
        "Height_cm": float(height),
        "Weight_kg": float(weight),
        "BMI": float(bmi_val),
        "Blood_Pressure_Systolic": float(bp_systolic),
        "Blood_Pressure_Diastolic": float(bp_diastolic),
        "Cholesterol_Level": float(cholesterol),
        "Blood_Sugar_Level": float(blood_sugar),
        "Daily_Steps": float(daily_steps),
        "Exercise_Frequency": exercise_freq, 
        "Sleep_Hours": float(sleep_hours),
        "Alcohol_Consumption": alcohol,
        "Smoking_Habit": smoking,
        "Dietary_Habits": dietary_habit
    }

    df_input = pd.DataFrame([input_data])
    
    feature_cols = schema_data['feature_cols']
    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = 0 
    df_input = df_input[feature_cols]

    cat_features_names = schema_data.get('categorical_cols', [])
    cat_features_indices = [df_input.columns.get_loc(c) for c in df_input.columns if c in cat_features_names]
    prediction_pool = Pool(df_input, cat_features=cat_features_indices)

    # 2. Run Predictions
    pred_calories = models['calories'].predict(prediction_pool)[0]
    pred_protein = models['protein'].predict(prediction_pool)[0]
    pred_carbs = models['carbs'].predict(prediction_pool)[0]
    pred_fats = models['fats'].predict(prediction_pool)[0]

    final_cal = int(max(1200, min(4500, pred_calories)))
    final_protein = int(max(10, pred_protein))
    final_carbs = int(max(10, pred_carbs))
    final_fats = int(max(10, pred_fats))

    # 3. Display Nutrition Metrics (Removed Meal Plan Metric)
    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
    col_res1.metric("Daily Calories", f"{final_cal} kcal")
    col_res2.metric("Target Protein", f"{final_protein}g")
    col_res3.metric("Target Carbs", f"{final_carbs}g")
    col_res4.metric("Target Fats", f"{final_fats}g")
    
    # 4. Integrated Workout Info Box
    routine_text = "General Strength & Cardio" # Fallback
    if "gym" in rec_resources and not rec_resources["gym"].empty:
        gym_df = rec_resources["gym"]
        gym_goal_map = {"Muscle Gain": "muscle_gain", "Fat Burn": "fat_burn"}
        target_goal = gym_goal_map.get(gym_goal, "muscle_gain")
        
        workout_match = gym_df[
            (gym_df['Goal'] == target_goal) &
            (gym_df['BMI Category'].str.lower() == bmi_cat.lower())
        ]
        if not workout_match.empty:
            routine_text = workout_match.iloc[0]['Exercise Schedule']

    st.info(f"""
    **Note:** For a goal of **{gym_goal}** with your BMI status ({bmi_cat}), we recommend:  
    **Exercise Schedule:** {routine_text}  
    **Target Intake:** {final_cal} kcal/day. Meals below are balanced to meet this target.
    """)

# 5. Generate Weekly Meal Plan
    st.subheader("🗓️ Your 7-Day Meal Plan")
    
    try:
        recommendations = get_knn_recommendations(
            final_cal, final_protein, final_fats, final_carbs, 
            rec_resources, dietary_habit, gym_goal
        )
        
        if recommendations is not None and not recommendations.empty:
            # Rename columns for better presentation
            recommendations = recommendations.rename(columns={
                "Meal": "Meal Type",
                "Name": "Recipe Name"
            })

            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            for day in days:
                day_data = recommendations[recommendations['Day'] == day].copy()
                
                if not day_data.empty:
                    with st.expander(f"**{day} View**", expanded=(day == "Monday")):
                        # Ensure Snack is between Lunch and Dinner
                        meal_order = ["Breakfast", "Lunch", "Snack", "Dinner"]
                        day_data['Meal Type'] = pd.Categorical(day_data['Meal Type'], categories=meal_order, ordered=True)
                        day_data = day_data.sort_values("Meal Type")

                        # Display with fixed column widths and hidden index
                        st.dataframe(
                            day_data[['Meal Type', 'Recipe Name', 'Calories', 'Protein', 'Carbs', 'Fats']],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Meal Type": st.column_config.TextColumn(
                                    "Meal Type",
                                    width="medium", # Sets a consistent uniform size for the first column
                                ),
                                "Recipe Name": st.column_config.TextColumn(
                                    "Recipe Name",
                                    width="large",
                                ),
                                "Calories": st.column_config.NumberColumn(format="%d kcal"),
                                "Protein": st.column_config.NumberColumn(format="%d g"),
                                "Carbs": st.column_config.NumberColumn(format="%d g"),
                                "Fats": st.column_config.NumberColumn(format="%d g"),
                            }
                        )
        else:
            st.warning("Could not generate recipes. Please check your data files.")
            
    except Exception as e:
        st.error(f"An error occurred during meal planning: {e}")