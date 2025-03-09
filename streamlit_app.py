import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Cache data loading to improve performance
@st.cache_data
def load_data(url):
    try:
        data = pd.read_csv(url)
        return data
    except Exception as e:
        st.error("Error loading data: " + str(e))
        return None

# ---------------------------
# Load and prepare data
DATA_URL = "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv"
df = load_data(DATA_URL)
if df is None:
    st.stop()

st.title('Machine Learning App')
st.info('RandomForestClassifier for Penguin species prediction.')

# Display raw data and features
with st.expander('Data'):
    st.write('**Raw Data**')
    st.dataframe(df)
    
    st.write('**Features (X)**')
    X_raw = df.drop('species', axis=1)
    st.dataframe(X_raw)
    
    st.write('**Target (y)**')
    y_raw = df['species']
    st.dataframe(y_raw)

# ---------------------------
# Data Visualization Section
with st.expander('Data Visualization'):
    st.write("### Scatter Chart: Bill Length vs Body Mass")
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')
    
    st.write("### Correlation Heatmap")
    # Select only numeric columns to avoid conversion errors
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ---------------------------
# Sidebar for Input Features and Model Parameters
with st.sidebar:
    st.header('Input Features')
    
    # Input feature widgets
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    bill_length_mm = st.slider("Bill Length (mm)", 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider("Bill Depth (mm)", 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider("Flipper Length (mm)", 172.0, 231.0, 201.0)
    # Fixed: default value set within the range (2700.0, 4207.0)
    body_mass_g = st.slider("Body Mass (g)", 2700.0, 4207.0, 3700.0)
    gender = st.selectbox('Gender', ('Male', 'Female'))

    # Create DataFrame for the input features
    input_data = {
        'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': gender
    }
    input_df = pd.DataFrame(input_data, index=[0])
    
    st.write("### Model Hyperparameters")
    n_estimators = st.slider("Number of Estimators", 10, 200, 100)
    max_depth = st.slider("Max Depth", 1, 20, 5)

# Display input features
with st.expander('Input Features'):
    st.write("**User Input**")
    st.dataframe(input_df)

# ---------------------------
# Data Preparation
# To avoid merging issues, we encode input data and training data separately using get_dummies
def prepare_features(df):
    # Create dummies for categorical features
    return pd.get_dummies(df, columns=['island', 'sex'])

# Encode training data (excluding the species column)
X_train = prepare_features(X_raw)

# Encode the input data using the same procedure
input_prepared = prepare_features(input_df)

# Align columns of input data with training data
input_prepared = input_prepared.reindex(columns=X_train.columns, fill_value=0)

# Encode target variable
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
y = y_raw.apply(lambda val: target_mapper[val])

with st.expander('Data Preparation'):
    st.write("**Encoded Training Features (X_train)**")
    st.dataframe(X_train.head())
    st.write("**Encoded User Input**")
    st.dataframe(input_prepared)
    st.write("**Encoded Target (y)**")
    st.dataframe(y.head())

# ---------------------------
# Model Training and Prediction
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
clf.fit(X_train, y)

# Make predictions
prediction = clf.predict(input_prepared)
prediction_proba = clf.predict_proba(input_prepared)

# Create a DataFrame for prediction probabilities
df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])

# Display predicted probabilities with progress bars
st.subheader('Predicted Species Probabilities')
st.dataframe(df_prediction_proba,
             column_config={
                 'Adelie': st.column_config.ProgressColumn(
                     'Adelie',
                     format='%f',
                     width='medium',
                     min_value=0,
                     max_value=1
                 ),
                 'Chinstrap': st.column_config.ProgressColumn(
                     'Chinstrap',
                     format='%f',
                     width='medium',
                     min_value=0,
                     max_value=1
                 ),
                 'Gentoo': st.column_config.ProgressColumn(
                     'Gentoo',
                     format='%f',
                     width='medium',
                     min_value=0,
                     max_value=1
                 ),
             }, hide_index=True)

# Map prediction back to species name
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
predicted_species = penguins_species[prediction][0]
st.success(f"Predicted Species: **{predicted_species}**")
