import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load model
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Streamlit UI
st.title("English → Hindi Translator")
st.write("Type a sentence in English and get the Hindi translation.")

# Input box
english_text = st.text_area("Enter English text:", height=100)

# Translate button
if st.button("Translate"):
    if english_text.strip() == "":
        st.warning("Please enter some text to translate!")
    else:
        # Encoding input and generating translation
        encoded = tokenizer(english_text, return_tensors="pt", padding=True)
        translated = model.generate(**encoded, max_length=50)
        hindi_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # Display output
        st.success("Hindi Translation:")
        st.write(hindi_text)
