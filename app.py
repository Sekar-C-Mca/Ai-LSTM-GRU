from pathlib import Path
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / 'lstm.h5'
TOKENIZER_PATH = PROJECT_ROOT / 'tokenizer.pickle'


st.set_page_config(page_title='Ai LSTM GRU', page_icon='🧠', layout='centered')


@st.cache_resource
def load_project_assets():
    ai_lstm_gru_model = load_model(MODEL_PATH)

    with TOKENIZER_PATH.open('rb') as handle:
        ai_lstm_gru_tokenizer = pickle.load(handle)

    return ai_lstm_gru_model, ai_lstm_gru_tokenizer


def predict_next_word(ai_lstm_gru_model, ai_lstm_gru_tokenizer, text, max_sequence_length):
    token_list = ai_lstm_gru_tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_length:
        token_list = token_list[-(max_sequence_length - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
    predicted = ai_lstm_gru_model.predict(token_list, verbose=0)
    predicted_word_index = int(np.argmax(predicted, axis=1)[0])

    return ai_lstm_gru_tokenizer.index_word.get(predicted_word_index)


st.title('Ai LSTM GRU Next Word Prediction')
st.write('Type a short sequence and the model will predict the next word.')

if not MODEL_PATH.exists():
    st.error(f'Model file not found: {MODEL_PATH.name}')
    st.stop()

if not TOKENIZER_PATH.exists() or TOKENIZER_PATH.stat().st_size == 0:
    st.error(f'Tokenizer file is missing or empty: {TOKENIZER_PATH.name}')
    st.stop()

ai_lstm_gru_model, ai_lstm_gru_tokenizer = load_project_assets()
input_text = st.text_input('Enter the sequence of words', 'To be or not to')

if st.button('Predict Next Word'):
    project_max_sequence_length = ai_lstm_gru_model.input_shape[1] + 1
    next_word = predict_next_word(
        ai_lstm_gru_model,
        ai_lstm_gru_tokenizer,
        input_text,
        project_max_sequence_length,
    )
    st.success(f'Next word: {next_word}' if next_word else 'No prediction available for that input.')