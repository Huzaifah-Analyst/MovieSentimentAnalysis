import streamlit as st
import pickle
import os
import numpy as np

# Set up the page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬"
)

# Add a title and description
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to analyze its sentiment")

# Load the model and tokenizer
@st.cache_resource
def load_model():
    # Get the current directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'sentiment_model.pkl')
    tokenizer_path = os.path.join(current_dir, 'tokenizer.pkl')
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at: {tokenizer_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def preprocess_text(text, tokenizer):
    """Handle different types of tokenizers"""
    try:
        # Try sklearn-style transform method
        if hasattr(tokenizer, 'transform'):
            return tokenizer.transform([text])
        
        # Try Keras/TensorFlow tokenizer
        elif hasattr(tokenizer, 'texts_to_sequences'):
            sequences = tokenizer.texts_to_sequences([text])
            # You might need to pad sequences if your model expects fixed length
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            # Adjust max_length based on your model's requirements
            max_length = 100  # Change this to match your model's expected input length
            return pad_sequences(sequences, maxlen=max_length)
        
        # Try if it's a simple vectorizer with fit_transform capability
        elif hasattr(tokenizer, 'fit_transform'):
            return tokenizer.transform([text])
        
        # If it's a custom tokenizer, try common methods
        elif hasattr(tokenizer, 'encode'):
            return tokenizer.encode([text])
        
        elif hasattr(tokenizer, '__call__'):
            return tokenizer([text])
        
        else:
            raise AttributeError("Tokenizer doesn't have a recognized method for text processing")
            
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        st.write(f"Tokenizer type: {type(tokenizer)}")
        st.write(f"Available methods: {[method for method in dir(tokenizer) if not method.startswith('_')]}")
        return None

try:
    model, tokenizer = load_model()
    
    # Create a text input for the user
    user_input = st.text_area("Enter your movie review:", height=150)
    
    if st.button("Analyze Sentiment"):
        if model is not None and tokenizer is not None:
            try:
                processed_input = preprocess_text(user_input, tokenizer)
                prediction = model.predict(processed_input)
                value = float(prediction[0]) if hasattr(prediction, '__getitem__') else float(prediction)
                sentiment = "Negative" if value < 0.5 else "Positive"
                color = "#ffcccc" if sentiment == "Negative" else "#ccffcc"
                st.markdown(f"""
                    <div style='background-color: {color}; padding: 20px; border-radius: 10px;'>
                        <h3 style='color: #333;'>Sentiment: {sentiment}</h3>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.error("Model or tokenizer not loaded.")
            st.warning("Please enter some text to analyze")

except FileNotFoundError as e:
    st.error(f"File not found: {str(e)}")
except Exception as e:
    st.error(f"Error loading the model or tokenizer: {str(e)}")
    st.write("Please make sure:")
    st.write("1. Both sentiment_model.pkl and tokenizer.pkl files are in the same directory as app.py")
    st.write("2. The files are not corrupted")
    st.write("3. You have the required dependencies installed (scikit-learn, tensorflow, etc.)")
    
    # Debug information
    try:
        model, tokenizer = load_model()
        st.write(f"Model type: {type(model)}")
        st.write(f"Tokenizer type: {type(tokenizer)}")
        st.write(f"Tokenizer methods: {[method for method in dir(tokenizer) if not method.startswith('_')]}")
    except:
        pass