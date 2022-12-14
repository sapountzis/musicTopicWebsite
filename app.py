import os

import joblib
import pandas as pd
import streamlit as st
import numpy as np
from top2vec import Top2Vec
import urllib.request
from pathlib import Path


@st.experimental_singleton
def get_data(data_local_path: str):
    return pd.read_feather(data_local_path)


@st.experimental_singleton
def get_lyrics_index(data):
    return pd.Index(data['lyrics'])


@st.experimental_singleton
def get_model(model_local_path: str):
    model: Top2Vec = joblib.load(model_local_path)
    model._check_model_status()
    return model


def get_similar(input: str, model: Top2Vec):
    vec = np.array(model.embed([input])[0])
    return model.search_documents_by_vector(vec, num_docs=20)[0]


def get_songs(data, lyrics_index, lyrics):
    return data.iloc[lyrics_index.get_indexer(lyrics)][['artist', 'song']].reset_index(drop=True)


if __name__ == "__main__":
    # MODEL_LOCAL_PATH = "models/top2vec-self.model"
    # DATA_LOCAL_PATH = "data/all_data_clean_corrected_english.feather"
    MODEL_LOCAL_PATH = "/content/drive/MyDrive/MusicTopics/top2vec-self.model"
    DATA_LOCAL_PATH = "/content/drive/MyDrive/MusicTopics/all_data_clean_corrected_english.feather"

    data = get_data(DATA_LOCAL_PATH)
    lyrics_index = get_lyrics_index(data)
    data = data.drop(columns=['lyrics'])
    model = get_model(MODEL_LOCAL_PATH)

    st.title('Music Topic Search')
    form = st.form('search')
    c = st.empty()
    with form:
        st.markdown('Search for music topics and get similar songs')
        search = st.text_input('Search for a topic', value='')
        submit = st.form_submit_button('Search')

        if submit:
            res = get_similar(search, model)
            res = get_songs(data, lyrics_index, res)
            res.index += 1

            with c:
                st.dataframe(res, height=750)

