English-Kannada Translation Project

This project explores multiple methods for translating English text into Kannada, ranging from classical APIs to modern deep learning models. We experimented with Google Translate and MyMemory API for quick translations, which worked, but required external services.

Next, we implemented an LSTM-based encoder-decoder model using a small dataset of English-Kannada sentence pairs. While this demonstrated the core concepts of sequence-to-sequence translation, the model struggled to produce accurate results due to limited training data and the complexity of the task.

Finally, we adopted a transformer-based MarianMT model (Helsinki-NLP), which provided reliable and accurate translations even with minimal setup. A simple Streamlit UI was built to allow real-time English-to-Kannada translation, making the project interactive and user-friendly. This journey showcases how modern NLP transformers can outperform traditional methods in language translation tasks.
