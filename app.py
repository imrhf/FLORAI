import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
st.set_option('deprecation.showPyplotGlobalUse', False)


st.set_page_config(page_title="FLORAI", layout="centered")


# Load dataset (replace "Crop_recommendation.csv" with your dataset)
crop_data = pd.read_csv("Crop_recommendation.csv")

df_boston = crop_data
df_boston.columns = df_boston.columns
df_boston.head()
# Function to preprocess the data
def preprocess_data(data):
    # Calculate IQR
    Q1 = np.percentile(data['rainfall'], 25, interpolation='midpoint')
    Q3 = np.percentile(data['rainfall'], 75, interpolation='midpoint')
    IQR = Q3 - Q1

    # Calculate upper and lower bounds
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR

    # Filter data based on bounds
    data = data[(data['rainfall'] >= lower_bound) & (data['rainfall'] <= upper_bound)]

    return data
crop_data = preprocess_data(crop_data)

goals = [
    {"title": "...", "description": "The proposal of the exact crop for a particular field area is the main goal. And by choosing the right crops we can reduce crop loss.",  "color": "#B3E5DB"},
    {"title": "...", "description": "Increase crop productivity, analyze crops in real-time, make smarter decisions and get better productivity.", "color": "#006657"},
    {"title": "...", "description": "Improve the quality of cultivation using the latest technologies.", "color": "#762927"}
]
# Welcome Page
def welcome_page():
    st.session_state.page_selection = "Welcome" 
    st.image("florailogo1.png",width= 400)
    st.markdown("<h5 style='color: #006657;'>A Good Crop Is The Result of Algorithms </h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #B3E5DB;'>We Help You Make Crop Recommendations Based On Soil Parameters And Environmental Conditions .</h5>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: dark gray;'OUR GOALS</h2>",unsafe_allow_html=True)
    st.markdown("<h3 style='color: #006657;'>OUR GOALS </h3>", unsafe_allow_html=True)



    # Initialize current_index
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    # Display current goal
    current_index = st.session_state.current_index
    goal = goals[current_index]

    st.markdown(f"""
    <div style="background: linear-gradient(to right, {goal['color']} 9%, transparent 70%); padding: 30px; border-radius: 15px; box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2); transition: 0.3s; margin-bottom: 20px;">
       <h1 style="color: white;">{goal['title']}</h1>
        <h4 style="color: black; font-size : 12 ;">{goal['description']}</h4>
    </div>
    """, unsafe_allow_html=True)

    # Navigation arrows
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("←", key="prev_button", help="Previous"):
            if current_index > 0:
                st.session_state.current_index -= 1
    with col3:
        if st.button("→", key="next_button", help="Next"):
            if current_index < len(goals) - 1:
                st.session_state.current_index += 1


# Input and Recommendation Page
def input_recommendation_page():
    st.title("Input and Recommendation")

    # User inputs
    st.header("Enter The Values")

    nitrogen = st.number_input("Nitrogen Content in Soil", min_value=0.0, step=0.1)
    phosphorus = st.number_input("Phosphorus Content in Soil", min_value=0.0, step=0.1)
    potassium = st.number_input("Potassium Content in Soil", min_value=0.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, step=1)
    pH = st.number_input("pH of Soil", min_value=0.0, max_value=14.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)

    # Algorithm selection
    algorithm = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "Naive Bayes", "KNN"])
    

    if algorithm:
        st.write(f"You selected {algorithm}.")
        st.write(f"Accuracy for {algorithm}: {acc[model.index(algorithm)]}")
    
    # Button to get recommendation
    if st.button("Get Recommendation", key="recommendation_button"):
        if algorithm:
            if algorithm == "Random Forest":
                prediction = rf.predict([[nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]])
            elif algorithm == "Naive Bayes":
                prediction = gnb.predict([[nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]])
            elif algorithm == "KNN":
                prediction = knn.predict([[nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]])
            st.success(f"Based on the inputs, the recommended crop is: {prediction[0]}")
        else:
            st.error("Please select an algorithm before getting recommendation.")

# Visualization Page
def visualization_page():
    st.title("Crop Parameter Visualization")

    st.subheader("Select a Parameter to Visualize")

    selected_parameter = st.selectbox("Parameter", crop_data.columns[:-1])

    st.subheader(f"Average {selected_parameter} Required for Each Crop")

    average_values = crop_data.groupby("label")[selected_parameter].mean()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=average_values.index, y=average_values)
    plt.xlabel("Crop")
    plt.ylabel(f"Average {selected_parameter}")
    plt.xticks(rotation=45)
    st.pyplot()


def main():
    
    # Call the preprocess_data function
    preprocess_data(crop_data)

    # Train models
    X = crop_data.drop(columns=['label'])
    y = crop_data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    global rf, gnb, knn
    
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_pred)
    
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb_pred = gnb.predict(X_test)
    gnb_acc = accuracy_score(y_test, gnb_pred)

    rf = RandomForestClassifier(n_estimators=20, random_state=0)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    global acc, model
    acc = [knn_acc, gnb_acc, rf_acc]
    model = ['KNN', 'Naive Bayes', 'Random Forest']
    pages = {
        "Welcome": welcome_page,
        "Input and Recommendation": input_recommendation_page,
        "Crop Parameter Visualization": visualization_page
    }
    

    st.sidebar.image("florailogo2.png" ,width = 300)
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    page = pages[selection]
    page()

if __name__ == "__main__":
    
    main()