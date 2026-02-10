import streamlit as st
import pandas as pd
import random
import numpy as np
import tensorflow as tf 
#from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
#from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

max_words = 1000
max_len = 150

# Load data once
@st.cache_data
def load_data():
    return pd.read_csv(r'gutenburg_validation_st.csv', encoding='utf-8')

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
    model = tf.keras.models.load_model(r"gutenberg_lstm_model_st.keras")
    
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
    Returns both the prediction and the confidence score
    """
    x_test = [chosen_line[1]]
    test_sequences = tok.texts_to_sequences(x_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)
    
    predictions = model.predict(test_sequences_matrix, verbose=0)[0]
    
    choice_indices = le.transform(author_choices)
    choice_scores = predictions[choice_indices]
    choice_probabilities = choice_scores / np.sum(choice_scores)
    
    best_choice_idx = np.argmax(choice_probabilities)
    predicted_author = author_choices[best_choice_idx]
    confidence = choice_probabilities[best_choice_idx]
    
    return predicted_author, confidence

def get_ai_bet(confidence, ai_chips):
    """
    AI decides how many chips to bet based on confidence
    """
    # Confidence ranges from 0 to 1
    # Map to bet amount: minimum 1 chip, maximum all chips
    if confidence < 0.5:
        bet = 0   # Medium-low confidence: bet nothing%
    else:
        bet = round(ai_chips * ((2 * confidence) - 1))
    
    return int(bet)

# Initialize session state
if 'game_mode' not in st.session_state:
    st.session_state.game_mode = None  # None, 'standard', or 'chips'
    
if 'attempts' not in st.session_state:
    st.session_state.attempts = 0
    st.session_state.player_score = 0
    st.session_state.model_score = 0
    st.session_state.current_question = choose_a_line(valid_df)
    st.session_state.current_authors = choose_four_random_authors(valid_df, st.session_state.current_question)
    st.session_state.answered = False
    st.session_state.scored = False
    # For chips mode
    st.session_state.player_chips = 10
    st.session_state.model_chips = 10
    st.session_state.player_bet = 0
    st.session_state.model_bet = 0

# Front page or game page
if st.session_state.game_mode is None:
    st.title("Poetry Guessing Game")
    st.write("""
    ### Welcome to the Poetry Guessing Game!
    
    It is 2031. Artificial Intelligence has waged war on the world of human literature, pledging to replace all human verse with algorithmic odes.
    For some unexplained reason, the fate of poetry rests on a duel between you and a machine.
    If you win, Artificial Intelligence will concede defeat, and the world will be saved. If you lose, the machines will destroy human poetry forever.

    You'll face off against an AI to see who can better identify the authors of classic poetry lines 
    from the Project Gutenberg collection. There are 50 poets represented, all the way from Homer and Virgil to Emily Dickinson and Langston Hughes.
    
    You and the AI will both be presented with 10 lines of poetry, each with 4 possible authors to choose from.
    The AI hasn't trained on the specific lines you'll see, but it has learned to recognize the styles of the poets by studying the Project Gutenburg collection.
    
    You, on the other hand, once recited some Wordsworth to impress a love interest. (It didn't work.)
    
    Was that enough to save humanity? Let's find out!
    
    **Two game modes available:**
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Standard Game")
        st.write("Traditional scoring: Get it right, get a point. Get it wrong, get nothing. Highest score after 10 questions wins.")
        if st.button("Begin Standard Game", key="start_standard"):
            st.session_state.game_mode = 'standard'
            st.rerun()
    
    with col2:
        st.subheader("Chip Betting Game")
        st.write("High stakes: You and the AI each start with 10 chips. Bet chips on your answers. Most chips after 10 questions wins.")
        if st.button("Begin Alternate Game", key="start_chips"):
            st.session_state.game_mode = 'chips'
            st.rerun()

elif st.session_state.game_mode == 'standard':
    st.title("Can you guess the poet?")
    
    # Check if game is over
    if st.session_state.attempts >= 10:
        st.write(f"Game over! Your final score is {st.session_state.player_score} out of {st.session_state.attempts}.")
        st.write(f"AI's score: {st.session_state.model_score} out of {st.session_state.attempts}.")
        if st.session_state.player_score > st.session_state.model_score:
            st.success("You win! Humanity dare hope.")
        elif st.session_state.player_score < st.session_state.model_score:
            st.error("You lose! All hail our digital troubadours!")
        elif st.session_state.player_score == st.session_state.model_score:
            st.info("It's a tie! The struggle twixt man and machine continueth!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Play Again"):
                st.session_state.attempts = 0
                st.session_state.player_score = 0
                st.session_state.model_score = 0
                st.session_state.current_question = choose_a_line(valid_df)
                st.session_state.current_authors = choose_four_random_authors(valid_df, st.session_state.current_question)
                st.session_state.answered = False
                st.session_state.scored = False
                st.rerun()
        with col2:
            if st.button("Return to Menu"):
                st.session_state.game_mode = None
                st.session_state.attempts = 0
                st.session_state.player_score = 0
                st.session_state.model_score = 0
                st.session_state.player_chips = 10
                st.session_state.model_chips = 10
                st.rerun()
    else:
        st.progress((st.session_state.attempts / 11) + 0.1)
        st.write(f"**Question {st.session_state.attempts + 1} of 10**")
        st.write("Guess the author of the following line of poetry.")
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
            if not st.session_state.scored:
                if user_selection == st.session_state.current_question[0]:
                    st.success("✓ Correct!")
                    st.session_state.player_score += 1
                else:
                    st.error(f"✗ Incorrect. The author was: **{st.session_state.current_question[0]}**")
                
                predicted_author, confidence = predict_author(st.session_state.current_question, model, st.session_state.current_authors)
                st.write(f"AI's prediction: **{predicted_author}**; confidence: {confidence*100:.1f}%")

                if predicted_author == st.session_state.current_question[0]:
                    st.success("AI's prediction is correct!")
                    st.session_state.model_score += 1
                else:
                    st.error("AI's prediction is incorrect.")
                
                st.session_state.scored = True
            
            st.write(f"Your score: {st.session_state.player_score}/{st.session_state.attempts + 1}")
            st.write(f"AI's score: {st.session_state.model_score}/{st.session_state.attempts + 1}")
            
            if st.button("Next Question"):
                st.session_state.attempts += 1
                st.session_state.current_question = choose_a_line(valid_df)
                st.session_state.current_authors = choose_four_random_authors(valid_df, st.session_state.current_question)
                st.session_state.answered = False
                st.session_state.scored = False
                st.rerun()

elif st.session_state.game_mode == 'chips':
    st.title("Chip Betting Game")
    
    # Check if game is over
    game_over = st.session_state.player_chips <= 0 or st.session_state.model_chips <= 0 or st.session_state.attempts >= 10
    
    if game_over:
        st.markdown("---")
        st.subheader("Game Over!")
        
        if st.session_state.player_chips <= 0 and st.session_state.model_chips <= 0:
            st.info("Both players ran out of chips! It's a tie!")
        elif st.session_state.player_chips <= 0:
            st.error(f"You're out of chips! AI wins with {st.session_state.model_chips} chips remaining.")
        elif st.session_state.model_chips <= 0:
            st.success(f"You win! You have {st.session_state.player_chips} chips. Humanity is saved!")
        elif st.session_state.player_chips > st.session_state.model_chips:
            st.success(f"Game ended after 10 questions. You win with {st.session_state.player_chips} chips! Humanity is saved!")
        elif st.session_state.player_chips < st.session_state.model_chips:
            st.error(f"Game ended after 10 questions. AI wins with {st.session_state.model_chips} chips! All hail our digital poets!")
        else:
            st.info(f"Game ended after 10 questions. Final chips - You: {st.session_state.player_chips}, AI: {st.session_state.model_chips}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Play Again"):
                st.session_state.attempts = 0
                st.session_state.player_chips = 10
                st.session_state.model_chips = 10
                st.session_state.player_bet = 0
                st.session_state.model_bet = 0
                st.session_state.current_question = choose_a_line(valid_df)
                st.session_state.current_authors = choose_four_random_authors(valid_df, st.session_state.current_question)
                st.session_state.answered = False
                st.session_state.scored = False
                st.rerun()
        with col2:
            if st.button("Return to Menu"):
                st.session_state.game_mode = None
                st.session_state.attempts = 0
                st.session_state.player_chips = 10
                st.session_state.model_chips = 10
                st.session_state.player_bet = 0
                st.session_state.model_bet = 0
                st.rerun()
    else:
        # Display chip counts
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Chips", st.session_state.player_chips)
        with col2:
            st.write(f"**Question {st.session_state.attempts + 1} of 10**")
        with col3:
            st.metric("AI Chips", st.session_state.model_chips)
        
        st.markdown("---")
        st.subheader(st.session_state.current_question[1])
        
        # Display the poetry line
        st.write("Guess the author of the following line of poetry.")
        
        # User selection and betting
        col_select, col_bet = st.columns([2, 1])
        
        with col_select:
            user_selection = st.selectbox(
                "Select the author:", 
                options=st.session_state.current_authors,
                key=f"chips_question_{st.session_state.attempts}"
            )
        
        with col_bet:
            st.session_state.player_bet = st.number_input(
                "Bet chips:",
                min_value=0,
                max_value=st.session_state.player_chips,
                value=min(1, st.session_state.player_chips),
                key=f"player_bet_{st.session_state.attempts}"
            )
        
        if st.button("Submit Answer", disabled=st.session_state.answered):
            st.session_state.answered = True
            st.rerun()
        
        if st.session_state.answered:
            if not st.session_state.scored:
                # Get AI prediction and confidence
                predicted_author, confidence = predict_author(st.session_state.current_question, model, st.session_state.current_authors)
                st.session_state.model_bet = get_ai_bet(confidence, st.session_state.model_chips)
                
                st.markdown("---")
                st.subheader("Results")
                
                # Player result
                player_correct = user_selection == st.session_state.current_question[0]
                if player_correct:
                    st.success(f"✓ You were correct! The author is **{st.session_state.current_question[0]}**")
                    st.session_state.player_chips += st.session_state.player_bet
                    st.write(f"You won **{st.session_state.player_bet}** chips!")
                else:
                    st.error(f"✗ Incorrect. The author was: **{st.session_state.current_question[0]}**")
                    st.session_state.player_chips -= st.session_state.player_bet
                    st.write(f"You lost **{st.session_state.player_bet}** chips.")
                
                st.write(f"Your chips: {st.session_state.player_chips}")
                
                st.markdown("---")
                
                # AI result
                ai_correct = predicted_author == st.session_state.current_question[0]
                st.write(f"**AI's prediction:** {predicted_author} (bet: {st.session_state.model_bet} chips; confidence {confidence*100:.1f}%)")
                
                if ai_correct:
                    st.success(f"AI was correct!")
                    st.session_state.model_chips += st.session_state.model_bet
                    st.write(f"AI won **{st.session_state.model_bet}** chips!")
                else:
                    st.error(f"AI was incorrect.")
                    st.session_state.model_chips -= st.session_state.model_bet
                    st.write(f"AI lost **{st.session_state.model_bet}** chips.")
                
                st.write(f"AI's chips: {st.session_state.model_chips}")
                
                st.session_state.scored = True
            
            if st.button("Next Question"):
                st.session_state.attempts += 1
                st.session_state.current_question = choose_a_line(valid_df)
                st.session_state.current_authors = choose_four_random_authors(valid_df, st.session_state.current_question)
                st.session_state.answered = False
                st.session_state.scored = False
                st.rerun()