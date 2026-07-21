import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load models
simple_model = load_model("simple_rnn.keras")
bi_model = load_model("bidirectional_rnn.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Stopwords and Lemmatizer
stop_words = set(stopwords.words("english"))

for word in ["no", "not", "nor"]:
    stop_words.discard(word)

lemmatizer = WordNetLemmatizer()


# Preprocessing function
def preprocess(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"#", "", text)

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"[^a-z'\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    words = []

    for word in text.split():

        if word not in stop_words:
            words.append(lemmatizer.lemmatize(word))

    return " ".join(words)


# ---------------- Streamlit UI ----------------

st.title("Twitter Sentiment Analysis")

st.write("Predict whether a tweet is Positive or Negative using RNN models.")

choice = st.selectbox(
    "Choose Model",
    ["Simple RNN", "Bidirectional RNN"]
)

tweet = st.text_area("Enter your tweet:")

if st.button("Predict"):

    if tweet.strip() == "":
        st.warning("Please enter a tweet.")

    else:

        clean = preprocess(tweet)

        seq = tokenizer.texts_to_sequences([clean])

        pad = pad_sequences(
            seq,
            maxlen=50,
            padding="post",
            truncating="post"
        )

        if choice == "Simple RNN":
            pred = simple_model.predict(pad, verbose=0)
        else:
            pred = bi_model.predict(pad, verbose=0)

        prediction = pred.argmax(axis=1)[0]

        label = encoder.inverse_transform([prediction])[0]

        confidence = pred.max() * 100

        st.subheader("Prediction")

        if label == "Positive":
            st.success(f"😊 {label}")
        else:
            st.error(f"😠 {label}")

        st.write(f"**Confidence:** {confidence:.2f}%")