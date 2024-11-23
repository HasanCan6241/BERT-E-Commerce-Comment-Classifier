import streamlit as st
import joblib
import numpy as np
import re
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

st.set_page_config(page_title="E-Ticaret Yorum Analiz Uygulaması", page_icon="🛒", layout="centered")


nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('turkish'))
lemmatizer = WordNetLemmatizer()

# Model ve tokenizer yükleme fonksiyonları
@st.cache_resource
def load_lstm_model():
    return load_model('lstm_model.keras')

@st.cache_resource
def load_tokenizer():
    return joblib.load('tokenizer.pkl')

@st.cache_resource
def load_label_encoder():
    return joblib.load('label_encoder.pkl')

lstm_model = load_lstm_model()
tokenizer = load_tokenizer()
label_encoder = load_label_encoder()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)

    text = re.sub(r'[^\w\s]', '', text)

    text = ' '.join([word for word in text.split() if word not in stop_words])

    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

st.title("🛍️ E-Ticaret Yorum Analiz Uygulaması")
st.write("Bu uygulama, e-ticaret sitelerinden alınan kullanıcı yorumlarını analiz eder ve hangi kategoriye ait olduğunu tahmin eder.")

user_input = st.text_area("Yorumunuzu girin:", height=150)

if st.button("Kategori Tahmin Et", key="predict_button"):
    if user_input:
        cleaned_comment = clean_text(user_input)

        new_sequence = tokenizer.texts_to_sequences([cleaned_comment])

        new_padded = pad_sequences(new_sequence, maxlen=200)

        predictions = lstm_model.predict(new_padded)

        predicted_classes = np.argmax(predictions, axis=1)

        predicted_category = label_encoder.inverse_transform(predicted_classes)

        st.success(f"Tahmin Edilen Kategori: **{predicted_category[0]}**", icon="✅")
    else:
        st.warning("Lütfen bir yorum girin.", icon="⚠")
