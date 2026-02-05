import streamlit as st
import pandas as pd
import random
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

max_words = 1000
max_len = 150

# Load data once
@st.cache_data
def load_data():
    return pd.read_csv(r'C:\Users\jezkn\Local\Data Science Projects\GutenbergPoetry\gutenburg_validation_st.csv', encoding='utf-8')

# Cache the model and encoders - this loads them only once!
@st.cache_resource
def load_model_and_encoders():
    valid_df = load_data()
    
    # Fit the tokenizer
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(valid_df['Line'])
    
    # Fit the label encoder
    le = LabelEncoder()
    le.fit(valid_df['Actual Author'])
    
    # Load the model
    model = tf.keras.models.load_model(r"C:\Users\jezkn\Local\Data Science Projects\GutenbergPoetry\gutenberg_lstm_model_st.keras")
    
    return model, tok, le

# Load everything once at startup
with st.spinner(text="Loading model..."):
    model, tok, le = load_model_and_encoders()

valid_df = load_data()

def choose_a_line(df):
    chosen = []
    selected_line = df.sample(n=1)
    true_author = selected_line['Actual Author'].iloc[0]
    true_line = selected_line['Line'].iloc[0]
    chosen.append(true_author)
    chosen.append(true_line)
    return chosen

def choose_four_random_authors(df, chosen_line):
    true_author = chosen_line[0]
    authors = df['Actual Author'].drop_duplicates()
    others = authors[authors != true_author].sample(n=3, replace=False).tolist()
    choices = others + [true_author]
    random.shuffle(choices)
    return choices

def predict_author(chosen_line, model, author_choices):
    """
    Predict author from the 4 given choices only
    """
    x_test = [chosen_line[1]]
    test_sequences = tok.texts_to_sequences(x_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)
    
    predictions = model.predict(test_sequences_matrix, verbose=0)[0]
    
    choice_indices = le.transform(author_choices)
    choice_probabilities = predictions[choice_indices]
    
    best_choice_idx = np.argmax(choice_probabilities)
    predicted_author = author_choices[best_choice_idx]
    
    return predicted_author

st.title("Can you guess the poet?")
st.write("Guess the author of the following line of poetry.")

# Initialize session state
if 'attempts' not in st.session_state:
    st.session_state.attempts = 0
    st.session_state.player_score = 0
    st.session_state.model_score = 0
    st.session_state.current_question = choose_a_line(valid_df)
    st.session_state.current_authors = choose_four_random_authors(valid_df, st.session_state.current_question)
    st.session_state.answered = False

# Check if game is over
if st.session_state.attempts >= 10:
    st.write(f"Game over! Your final score is {st.session_state.player_score} out of {st.session_state.attempts}.")
    st.write(f"Model's score: {st.session_state.model_score} out of {st.session_state.attempts}.")
    if st.button("Play Again"):
        st.session_state.attempts = 0
        st.session_state.player_score = 0
        st.session_state.model_score = 0
        st.session_state.current_question = choose_a_line(valid_df)
        st.session_state.current_authors = choose_four_random_authors(valid_df, st.session_state.current_question)
        st.session_state.answered = False
        st.rerun()
else:
    st.progress((st.session_state.attempts / 11) + 0.1)
    st.write(f"**Question {st.session_state.attempts + 1} of 10**")
    st.subheader(st.session_state.current_question[1])
    
    user_selection = st.selectbox(
        "Select the author you think wrote this line:", 
        options=st.session_state.current_authors,
        key=f"question_{st.session_state.attempts}"
    )
    
    if st.button("Submit Answer", disabled=st.session_state.answered):
        st.session_state.answered = True
        st.rerun()
    
    if st.session_state.answered:
        if user_selection == st.session_state.current_question[0]:
            st.success("✓ Correct!")
            st.session_state.player_score += 1
        else:
            st.error(f"✗ Incorrect. The author was: **{st.session_state.current_question[0]}**")
        
        predicted_author = predict_author(st.session_state.current_question, model, st.session_state.current_authors)
        st.write(f"Model's prediction: **{predicted_author}**")

        if predicted_author == st.session_state.current_question[0]:
            st.success("Model's prediction is correct!")
            st.session_state.model_score += 1
        else:
            st.error("Model's prediction is incorrect.")

        st.write(f"Your score: {st.session_state.player_score}/{st.session_state.attempts + 1}")
        st.write(f"Model's score: {st.session_state.model_score}/{st.session_state.attempts + 1}")
        
        if st.button("Next Question"):
            st.session_state.attempts += 1
            st.session_state.current_question = choose_a_line(valid_df)
            st.session_state.current_authors = choose_four_random_authors(valid_df, st.session_state.current_question)
            st.session_state.answered = False
            st.rerun()