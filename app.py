import pandas as pd
import streamlit as st
import numpy as np
from top2vec import Top2Vec
import urllib.request
from pathlib import Path


def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


@st.experimental_singleton
def get_data():
    return pd.read_feather('data/all_data_clean_corrected_english.feather')


@st.experimental_singleton
def get_model():
    model: Top2Vec = Top2Vec.load('models/top2vec-self.model')
    model._check_model_status()
    return model


def get_similar(input: str, model: Top2Vec):
    vec = np.array(model.embed([input])[0])
    return model.search_documents_by_vector(vec, num_docs=20)[0]


def get_songs(data: pd.DataFrame, lyrics: list[str]):
    return data[data['lyrics'].isin(lyrics)][['song', 'artist']].reset_index(drop=True)


if __name__ == "__main__":
    HERE = Path(__file__).parent
    MODEL_URL = "https://store1.gofile.io/download/a18d8fd2-ac18-4914-9b91-34265fce0d42/top2vec-self.model"
    DATA_URL = "https://store4.gofile.io/download/00311955-ebf4-4430-8ed2-656b5fbcf12b/all_data_clean_corrected_english.feather"
    MODEL_LOCAL_PATH = HERE / "models/top2vec-self.model"
    DATA_LOCAL_PATH = HERE / "data/all_data_clean_corrected_english.feather"

    download_file(MODEL_URL, MODEL_LOCAL_PATH)
    download_file(DATA_URL, DATA_LOCAL_PATH)

    model = get_model()
    data = get_data()
    st.title('Music Topic Search')
    c = st.empty()
    with st.sidebar:
        with st.form('search'):
            st.text('Search for music topics and get similar songs')
            search = st.text_input('Search for a topic', value='')
            submit = st.form_submit_button('Submit')

            if submit:
                res = get_similar(search, model)
                res = get_songs(data, res)
                res.index += 1

                with c:
                    st.dataframe(res, height=750)
                # st.dataframe(res, width=1000)

