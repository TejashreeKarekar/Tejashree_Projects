import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.decomposition import PCA



# Set the Streamlit page configuration
st.set_page_config(page_title="Model Forge:The Data Science Engine", page_icon="📊", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
   /* Paragraph and other text styling */
p, label {
    font-size: 16px;
    color: #dcdcdc; /* Light gray for body text */
    line-height: 1.5;
}

/* Button styling */
.stButton>button {
    background-color: #1e2130; /* White background for buttons */
    color: #000000; /* Black text for button label */
    border-radius: 12px;
    border: 2px solid #ffffff; /* White border for better button definition */
    width: 100%;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    margin-top: 20px;
    transition: background-color 0.3s ease;
}

/* Button hover effect */
.stButton>button:hover {
    background-color: #32364d; /* Light gray background on hover */
    cursor: pointer;
}

/* Input field styling */
.stTextInput>div>div>input {
    border: 2px solid #ffffff; /* White border for consistency */
    background-color: #333333; /* Dark gray background for input fields */
    color: #ffffff; /* White text for input fields */
    border-radius: 12px;
    padding: 12px;
    width: 100%;
    font-size: 16px;
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background: #1c1c1c; /* Dark gray background for sidebar */
    color: #ffffff; /* White text for visibility */
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}

/* Sidebar header styling */
.sidebar h2 {
    color: #ffffff; /* White for sidebar headers */
    font-weight: bold;
}

/* Block container padding */
.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Style for DataFrame display */
.stDataFrame {
    border-radius: 10px;
    border: 2px solid #ffffff; /* White border for consistency */
    margin-top: 20px;
}

/* Styling for section dividers */
hr {
    border: 0;
    height: 2px;
    background: #ffffff; /* White color for section dividers */
    margin-top: 30px;
    margin-bottom: 30px;
}

/* Custom styles for upload area */
.stFileUploader {
    background-color: #333333; /* Dark gray for upload area */
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 20px;
}

/* Custom styling for visual emphasis */
.highlight-text {
    background-color: #555555; /* Darker gray for emphasis */
    color: #ffffff; /* White text for better readability */
    padding: 5px 10px;
    border-radius: 5px;
    font-weight: bold;
}

</style>


""", unsafe_allow_html=True)

# Main title
st.title("Model Forge: The Ultimate Data Science Engine 📊")
st.write("Upload a CSV file to explore and analyze your dataset effortlessly.")

# Sidebar
st.sidebar.title("📌Navigation")
st.sidebar.write("Analyse your dataset!!! 😎")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Sidebar styling
with st.sidebar:
    st.markdown("---")
    st.markdown("Use the options to explore the functionalities!!")

# Function to load the dataset
def load_dataset(uploaded_file):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully!")
    st.write("**Dataset Preview**")
    st.dataframe(df.head())
    return df

# Function to display summary statistics
def summary_statistics(df):
    """Display summary statistics of the dataset."""
    st.subheader("📋 Summary Statistics")
    st.write("**Statistical Overview**")
    st.write(df.describe())
    st.write("**Data Types**")
    st.write(df.dtypes)
# Function to convert categorical columns to numerical
def convert_to_numerical(df):
    st.subheader("🔄 Convert Categorical to Numerical")
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.write(f"Categorical columns detected: {list(categorical_cols)}")
        if st.button("Convert to Numerical"):
            df_encoded = pd.get_dummies(df, columns=categorical_cols)
            st.write("Converted DataFrame:")
            st.dataframe(df_encoded)
            return df_encoded
    else:
        st.write("No categorical columns to convert.")
    return df

# Function to handle class imbalance using SMOTE
from sklearn.utils.multiclass import type_of_target

def handle_imbalance(df, target_column):
    from imblearn.over_sampling import SMOTE

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check if the target variable is suitable for classification
    target_type = type_of_target(y)
    if target_type not in ['binary', 'multiclass']:
        st.warning("The target variable is continuous. Consider converting it to categorical values if this is a classification task.")
        return df  # Return the original dataframe without applying SMOTE

    # If suitable for classification, proceed with SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Combine resampled features and target into a new DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = y_resampled

    return df_resampled


# Function for feature selection
def feature_selection(df):
    st.subheader("📊 Feature Selection")
    from sklearn.feature_selection import SelectKBest, chi2
    if df.select_dtypes(include=[np.number]).shape[1] > 1:
        num_features = st.slider("Select number of features to keep", 1, len(df.columns)-1, 5)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        selector = SelectKBest(score_func=chi2, k=num_features)
        X_new = selector.fit_transform(X, y)
        st.write("Top selected features:")
        st.dataframe(X_new)
        return X_new
    else:
        st.write("Not enough numeric features for selection.")
    return df

# Function for dimensionality reduction
def dimensionality_reduction(df):
    # Assuming 'Species' is your categorical target variable
    if 'Species' in df.columns:
        target = df['Species']
        df = df.drop('Species', axis=1)  # Drop target variable

    # Encode categorical features
    df = pd.get_dummies(df, drop_first=True)  # One-hot encode categorical variables

    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Apply PCA
    pca = PCA(n_components=2)  # Adjust the number of components as needed
    df_reduced = pca.fit_transform(df_scaled)

    # Convert to DataFrame for easier handling
    df_reduced = pd.DataFrame(data=df_reduced, columns=['Principal Component 1', 'Principal Component 2'])
    
    if 'Species' in locals():  # If target variable was present
        df_reduced['Species'] = target.reset_index(drop=True)  # Add target back to the DataFrame

    return df_reduced

# Function to analyze and visualize missing values
def analyze_missing_values(df):
    """Analyze and visualize missing values in the dataset."""
    st.subheader("🔍 Missing Values Analysis")

    # Check for missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()

    if total_missing == 0:
        st.write("There are no missing values in the dataset.")
        return

    st.write("**Missing Values in Each Column**")
    st.write(missing_values)

    # Heatmap for missing values
    st.write("**Missing Values Heatmap**")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu", linecolor='white', linewidths=0.5)
    plt.title("Missing Values Heatmap", fontsize=14)
    st.pyplot(plt)

# Function to detect and visualize outliers
def detect_outliers(df, column):
    """Detect outliers in a specific column using the IQR method."""
    st.subheader(f"📉 Outlier Detection: {column}")
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]

    st.write(f"**Number of Outliers Detected in {column}:** {len(outliers)}")
    if len(outliers) > 0:
        st.dataframe(outliers)
    else:
        st.write("No outliers detected.")

    # Visualize the outliers using a box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column], color="lightcoral")
    plt.title(f"Box Plot of {column}", fontsize=14)
    st.pyplot(plt)

# Function to generate basic visualizations
def visualize_data(df):
    """Generate basic visualizations for the dataset."""
    st.subheader("📊 Data Visualizations")

    # Histograms for numerical features
    st.write("### Histograms of Numerical Features")
    df.hist(figsize=(12, 8), bins=30, color='#3498DB', zorder=2, rwidth=0.9)
    plt.suptitle("Histograms", fontsize=14)
    st.pyplot(plt)

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    # Select only numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap", fontsize=14)
        st.pyplot(plt)
    else:
        st.write("No numeric columns available for correlation heatmap.")

# Load the dataset if a file is uploaded
if uploaded_file is not None:
    df = load_dataset(uploaded_file)

        # Add more options for new functionalities
    options = st.sidebar.radio(
        "Select an Analysis Step",
        ('Summary Statistics', 'Missing Values', 'Outlier Detection', 'Visualize Data',
         'Convert to Numerical', 'Handle Imbalance', 'Feature Selection', 'Dimensionality Reduction')
    )

    # Perform actions based on user selection
    if options == 'Summary Statistics':
        summary_statistics(df)
    elif options == 'Missing Values':
        analyze_missing_values(df)
    elif options == 'Outlier Detection':
        # Outlier detection for each numeric column
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            detect_outliers(df, column)
    elif options == 'Visualize Data':
        visualize_data(df)
    elif options == 'Convert to Numerical':
        df = convert_to_numerical(df)
    elif options == 'Handle Imbalance':
        target_column = st.sidebar.selectbox("Select Target Column", df.columns)
        df = handle_imbalance(df, target_column)
    elif options == 'Feature Selection':
        feature_selection(df)
    elif options == 'Dimensionality Reduction':
        dimensionality_reduction(df)

else:
    st.write("⬆️ Please upload a CSV file to proceed.")
