# Library Imports
import streamlit as st  # To create the application interface
import joblib   # To load models and vectors
import re   # To use regular expressions in text processing
import nltk # Natural Language Toolkit for text processing
from nltk.corpus import stopwords   # To access stopwords
from nltk.stem import PorterStemmer # For word stemming
from nltk.tokenize import word_tokenize # To convert words into tokens
import numpy as np  # For numerical operations

# Download NLTK data
# Check if NLTK tokenizer data is avialable, download if not found
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
# Check if NLTK stopwords data is avialable, download if not found
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set up page configuration
st.set_page_config(
    page_title="News Classification",   # Title shown in browser tab
    page_icon="",   # Set no icon
    layout="wide",  # Set layout
    initial_sidebar_state="expanded"    
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .prediction-real {
        font-size: 1.8rem;
        color: #4CAF50;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        background-color: #E8F5E9;
        text-align: center;
    }
    .prediction-fake {
        font-size: 1.8rem;
        color: #F44336;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        background-color: #FFEBEE;
        text-align: center;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #2196F3;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        width: 100%;
    }
    .sample-box {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #9CA3AF;
        margin-bottom: 15px;
        cursor: pointer;
    }
    .sample-box:hover {
        background-color: #E5E7EB;
    }
    </style>
    """, unsafe_allow_html=True) # To allow HTML rendering in streamlit

# Load the model with caching to improve performance
@st.cache_resource  # Decorator to cache the resource and avoid reloading on every interaction
def load_model():
    """
    Load the pre-trained model from pkl file
    
    The function attempts to load a trained model saved as a pickel file
    It uses caching to improve performance by avoiding reloading on each interaction
    
    Returns:
        model: The loaded machine learning model object
        None: If the model file is not found or an error occurs
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: For any other errors during model Loading
    """
    try:
        # Try to load the model using joblib
        model = joblib.load("random_forest_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file 'random_forest_model.pkl' not found. Please make sure it's in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
        return None

# Try to load TF-IDF vectorizer if it exists
@st.cache_resource  # Decorator to cache the resources
def load_vectorizer():
    """
    Load the TF-IDF vectorizer froma file or create a basic one if not found
    
    This function attemps to load a pre-fitted TF-IDF vectorizer, if the file is not found, it creates a basic vectorizer as a fallback option
    
    Returns:
        vectorizer: The loaded or newly created TF-IDF vectorizer object
    """
    try:
        # Check if a vectorizer file exists
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return vectorizer
    except:
        st.warning("TF-IDF vectorizer not found. Creating a basic one. For accurate results, please ensure you have the correct vectorizer.")
        # Create a basic vectorizer as fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(max_features=5000)

# Text preprocessing function
def preprocess_text(text):
    """
    Preprocess and clean text data before passing to the model
    
    This function performs NLP preprocessing steps:
    1. Converts text to lowercase
    2. Removes special characters and digits
    3. Tokenizes the text into words
    4. Removes stopwords and short words
    5. Applies stemming to reduce words to their root form
    
    Args:
        text(str): The raw text input to be processed
    
    Returns:
        str: The cleaned and processed text as a single string
    """
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

def predict_news(text):
    """
    Predict whether a news ariticle is real or fake
    
    This function takes raw text input, preprocesses it, vectorizes it, and uses the trained model to make a prediction about its authenticity.
    
    Args:
        text(str): The news text to be analyzed
        
    Returns:
        tuple: A tuple containing:
                - result (str): "Real News" or "Fake News"
                - confidence (float): Confidence score for the prediction
                - probabilities (array): Array of probablilities for both classes
        
    The model is a binary classification model
    0 = Fake News, 1 = Real News
    """
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    
    # Load vectorizer
    vectorizer = load_vectorizer()

    # Load Model
    model = load_model()
    
    # Transform using the vectorizer
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    # Prediction Results
    result = "Real News" if prediction == 1 else "Fake News"
    confidence = probability[1] if prediction == 1 else probability[0]
    
    return result, confidence, probability

# Main application
def main():
    """
    Main function
    
    This function:
    1. Sets up the application interface
    2. Loads the model and vectorizer
    3. Creates the user interface layout
    4. Handle user interactions
    5. Displays prediction results
    
    The application used stramlit for the front-end
    """

    # Display the main Header 
    st.markdown('<h1 class="main-header">News Classification</h1>', unsafe_allow_html=True)


    # Initialize session state for text input if it doesn;t exist 
    # Session state preserves values across reruns
    if "news_input" not in st.session_state:
        st.session_state.news_input = ""    # Initialize with empty string

    # Sample texts for demonstation
    
    # Real news example
    real_news_1 = """(Sourced From MalayMail 09/10/2025) From today onwards, Malaysia will no longer punish those who try to commit suicide with jail or fine, as it is no longer a crime in the country. Instead, the Malaysian government has improved its mental health law, by empowering more officers to rescue those who attempt suicide and quickly get medical help for them. In a joint statement by the Prime Minister’s Department’s Legal Affairs Division (BHEUU) and the Health Ministry (MOH), the government said it had today enforced three new laws in conjunction with World Suicide Prevention Day 2025. These three laws were passed in Parliament in 2023, but did not take effect until today. “It is the Madani government’s hope that the reforms of these laws will be a huge shift in efforts to prevent attempted suicides in Malaysia, by encouraging those whose mental health are affected, to step forward to get help; to eradicate stigma towards attempted suicides and reduce the rate of deaths due to suicides,” BHEUU and MOH said in the statement."""

    # Fake news example
    fake_news = """BREAKING: Government Secretly Installing Mind Control Devices in COVID Vaccines. In a shocking revelation, anonymous sources within the Pentagon have confirmed that the government is using COVID-19 vaccines to implant microscopic tracking and mind control devices in citizens. These nano-chips, developed by Bill Gates and funded by global elites, can monitor your thoughts and movements 24/7. The devices are activated by 5G towers that have been strategically placed across the country. People who received the vaccine report strange dreams and sudden urges to obey government mandates. One victim reported, After my second dose, I suddenly wanted to eat more vegetables and exercise daily - something I never did before! Doctors who have spoken out against this conspiracy have mysteriously disappeared. The mainstream media is covering up this scandal despite overwhelming evidence. Protect yourself by refusing vaccination and shielding your home with aluminum foil to block 5G signals.Share this urgent news before it gets censored! The truth must be revealed!"""

    # Define call back function to set sample text
    def set_sample(text):
        # Update the session state with the sample text
        st.session_state.news_input = text

    # Create a two-column loyout for the interface
    col1, col2 = st.columns([2, 1]) # First column is twice as wide as the second

    # Content for the first column
    # Contains the text box and button
    with col1:  
        # Heading
        st.markdown('<h3 class="sub-header">Enter News Text</h3>', unsafe_allow_html=True)
        
        # Text area for user input with initial value from session state
        input_text = st.text_area(
            "Paste the news article or headline here:",
            height=200,
            placeholder="Type or paste news content here...",
            key="news_input"
        )

        # Button to trigger detection
        detect_button = st.button("Detect", type="primary")

    # Content for second columns
    # displays information about the model
    with col2:
        # Header
        st.markdown('<h3 class="sub-header">About the Model</h3>', unsafe_allow_html=True)
        st.write("""
        This fake news detection model uses:
        - **Random Forest** algorithm  
        - **TF-IDF** for text vectorization  
        - **NLP preprocessing**  
        """)
        st.markdown("---")
        st.markdown("**Tips:** Use full articles for better accuracy.")

    # Section for sample news examples
    st.markdown("---")  # Horizontal divider
    st.markdown('<h3 class="sub-header">Try These Examples</h3>', unsafe_allow_html=True) # Header
    
    # Creates two columns for the example button
    sample_col1, sample_col2= st.columns(2)

    # Button for real News example
    with sample_col1:
        st.button("Real News Example 1", key="sample1", on_click=set_sample, args=(real_news_1,))
    
    # Button for fake news example
    with sample_col2:
        st.button("Fake News Example", key="sample2", on_click=set_sample, args=(fake_news,))
 

    # Run detection when the detect button is clicked
    if detect_button:

        # Check if input text is not empty
        if not input_text.strip():
            st.warning("Please enter some text to analyze.")    # Show warning if empty
        else:
            # Show spinner while processing
            with st.spinner("Analyzing text..."):
                try:
                    # Get prediction results
                    result, confidence, probabilities = predict_news(input_text)
                    st.markdown("---") # Horizaontal divider

                    # Display results header
                    st.markdown('<h3 class="sub-header">Detection Results</h3>', unsafe_allow_html=True)

                    # Display appropriate result based on prediction
                    if result == "Fake News":
                        st.markdown('<p class="prediction-fake">This news is likely FAKE</p>', unsafe_allow_html=True)
                        st.write(f"Confidence: {probabilities[0]*100:.2f}%")
                    else:
                        st.markdown('<p class="prediction-real">This news is likely REAL</p>', unsafe_allow_html=True)
                        st.write(f"Confidence: {probabilities[1]*100:.2f}%")
                # Handle exceptions
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")

# Entry point of the app
if __name__ == "__main__":
    main()