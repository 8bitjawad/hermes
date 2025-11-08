import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch

@st.cache_resource(show_spinner="Loading translation model...")
def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    try:
        # Load tokenizer and model
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.eval()  # Set the model to evaluation mode
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Streamlit UI
st.title("English → Hindi Translator")
st.write("Type a sentence in English and get the Hindi translation.")

# Load model
tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.error("Failed to load the translation model. Please check the logs for details.")
    st.stop()

# Input box
english_text = st.text_area("Enter English text:", height=100)

# Translate button
if st.button("Translate"):
    if not english_text.strip():
        st.warning("Please enter some text to translate!")
    else:
        # Encode the input text
        inputs = tokenizer(english_text, return_tensors="pt", padding=True, truncation=True)
        
        # Generate translation
        with torch.no_grad():
            translated = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
        # Decode the generated tokens to text
        hindi_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # Display output
        st.success("Hindi Translation:")
        st.write(hindi_text)
