import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import json
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

###################################################
# Load Vocabulary
###################################################

with open("models/vocab.json") as f:
    word2idx = json.load(f)

embedding_matrix = np.load("models/embedding_matrix.npy")

MAX_LEN = 60

###################################################
# Preprocess
###################################################

stop_words = set(stopwords.words("english"))

negation_words = {
    "no","nor","not",
    "don't","didn't","doesn't",
    "isn't","aren't","wasn't","weren't",
    "haven't","hasn't","hadn't",
    "won't","wouldn't","can't","couldn't",
    "shouldn't","mustn't",
    "never"
}

stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()


def preprocess(text):

    text = text.lower()

    contractions = {
        "can't":"can not",
        "won't":"will not",
        "n't":" not",
        "'re":" are",
        "'ve":" have",
        "'ll":" will",
        "'d":" would",
        "'m":" am",
        "'s":" is"
    }

    for k,v in contractions.items():
        text=text.replace(k,v)

    text=re.sub(r"http\S+|www\S+"," ",text)
    text=re.sub(r"<.*?>"," ",text)
    text=re.sub(r"\d+"," ",text)
    text=re.sub(r"[^a-z\s]"," ",text)
    text=re.sub(r"\s+"," ",text).strip()

    tokens=text.split()

    tokens=[
        w for w in tokens
        if w not in stop_words
    ]

    tokens=[
        lemmatizer.lemmatize(w)
        for w in tokens
    ]

    return tokens


def text_to_indices(tokens):

    ids=[
        word2idx.get(tok,1)
        for tok in tokens
    ]

    if len(ids)<MAX_LEN:
        ids += [0]*(MAX_LEN-len(ids))
    else:
        ids=ids[:MAX_LEN]

    length=min(len(tokens),MAX_LEN)

    return ids,length


###################################################
# Model
###################################################

class MultiTaskBiRNN(nn.Module):

    def __init__(self,vocab_size,embed_dim,embedding_matrix):

        super().__init__()

        self.embedding=nn.Embedding.from_pretrained(
            torch.tensor(
                embedding_matrix,
                dtype=torch.float
            ),
            freeze=False,
            padding_idx=0
        )

        self.birnn=nn.RNN(
            embed_dim,
            128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.dropout=nn.Dropout(0.4)

        self.sst_head=nn.Linear(256,2)
        self.mrpc_head=nn.Linear(256,2)
        self.qqp_head=nn.Linear(256,2)
        self.mnli_head=nn.Linear(256,3)

    def forward(self,input_ids,lengths,task):

        x=self.embedding(input_ids)

        packed=nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _,hidden=self.birnn(packed)

        h=torch.cat(
            (hidden[-2],hidden[-1]),
            dim=1
        )

        h=self.dropout(h)

        if task==0:
            return self.sst_head(h)

        if task==1:
            return self.mrpc_head(h)

        if task==2:
            return self.qqp_head(h)

        return self.mnli_head(h)


###################################################
# Load model
###################################################

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=MultiTaskBiRNN(
    len(word2idx),
    embedding_matrix.shape[1],
    embedding_matrix
)

model.load_state_dict(
    torch.load(
        "models/best_multitask_model.pt",
        map_location=device
    )
)

model.to(device)

model.eval()

###################################################
# Streamlit
###################################################

st.title("MultiTask NLP")

task=st.selectbox(
    "Task",
    [
        "SST-2",
        "MRPC",
        "QQP",
        "MNLI"
    ]
)

if task=="SST-2":

    sentence=st.text_area("Sentence")

    if st.button("Predict"):

        tokens=preprocess(sentence)

        ids,length=text_to_indices(tokens)

        ids=torch.tensor([ids]).to(device)
        length=torch.tensor([length]).to(device)

        with torch.no_grad():

            logits=model(ids,length,0)

            pred=torch.argmax(logits).item()

        st.success(
            "Positive" if pred else "Negative"
        )

elif task=="MRPC":

    s1=st.text_area("Sentence 1")

    s2=st.text_area("Sentence 2")

    if st.button("Predict"):

        tokens=preprocess(s1)+["<SEP>"]+preprocess(s2)

        ids,length=text_to_indices(tokens)

        ids=torch.tensor([ids]).to(device)
        length=torch.tensor([length]).to(device)

        with torch.no_grad():

            logits=model(ids,length,1)

            pred=torch.argmax(logits).item()

        st.success(
            "Paraphrase" if pred else "Not Paraphrase"
        )

elif task=="QQP":

    q1=st.text_area("Question 1")

    q2=st.text_area("Question 2")

    if st.button("Predict"):

        tokens=preprocess(q1)+["<SEP>"]+preprocess(q2)

        ids,length=text_to_indices(tokens)

        ids=torch.tensor([ids]).to(device)
        length=torch.tensor([length]).to(device)

        with torch.no_grad():

            logits=model(ids,length,2)

            pred=torch.argmax(logits).item()

        st.success(
            "Duplicate" if pred else "Different"
        )

else:

    p=st.text_area("Premise")

    h=st.text_area("Hypothesis")

    if st.button("Predict"):

        tokens=preprocess(p)+["<SEP>"]+preprocess(h)

        ids,length=text_to_indices(tokens)

        ids=torch.tensor([ids]).to(device)
        length=torch.tensor([length]).to(device)

        with torch.no_grad():

            logits=model(ids,length,3)

            pred=torch.argmax(logits).item()

        labels={
            0:"Entailment",
            1:"Neutral",
            2:"Contradiction"
        }

        st.success(labels[pred])