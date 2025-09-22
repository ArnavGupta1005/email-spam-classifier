#Importing Necessary files
import streamlit as st
import joblib
import pandas as pd

#Page Configuration
st.set_page_config(
    page_title="Email Spam Verifier",
    page_icon="📧",
    layout="wide"
)

#Load Model and Assets
@st.cache_resource
def load_model():
    """Load the pre-trained pipeline."""
    try:
        pipeline = joblib.load('spam_classifier_pipeline.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("Model file not found! Please run train_model.py first to generate the assets.")
        return None

pipeline = load_model()

#Model Explainability Function
@st.cache_data
def get_important_words(_pipeline, text, top_n=5):
    """Extract and rank words that contribute most to the spam/ham prediction."""
    if _pipeline is None:
        return [], []
        
    vectorizer = _pipeline.named_steps['tfidf']
    model = _pipeline.named_steps['model']
    feature_names = vectorizer.get_feature_names_out()
    spam_log_probs = model.feature_log_prob_[1]
    word_coeffs = dict(zip(feature_names, spam_log_probs))
    input_words = [word for word in text.lower().split() if word in word_coeffs]
    if not input_words:
        return [], []
        
    input_word_coeffs = {word: word_coeffs[word] for word in set(input_words)}
    spammy_words = sorted(input_word_coeffs.items(), key=lambda item: item[1], reverse=True)
    hammy_words = sorted(input_word_coeffs.items(), key=lambda item: item[1])
    
    return spammy_words[:top_n], hammy_words[:top_n]

#UI Layout
st.title("Email Spam Verifier 📧")
st.markdown("This tool uses a Naive Bayes model to predict whether an email is spam or not. Explore the tabs below to see the model in action, check its performance, and gain insights from the data.")

if pipeline is not None:
    #Tabbed Interface
    tab1, tab2, tab3 = st.tabs(["🔎 Live Prediction", "📊 Model Performance", "💡 Dataset Insights"])

    #Tab 1: Live Prediction
    with tab1:
        st.header("Check Your Email")
        user_input = st.text_area("Paste the email text here:", height=250, placeholder="Enter email content...")
        
        if st.button("Analyze Email", type="primary", use_container_width=True):
            if user_input:
                prediction = pipeline.predict([user_input])[0]
                probability = pipeline.predict_proba([user_input])
                
                st.write("---")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Prediction:")
                    if prediction == 'spam':
                        spam_prob = probability[0][1] * 100
                        st.error(f"Spam ({spam_prob:.2f}%)", icon="🚨")
                    else:
                        ham_prob = probability[0][0] * 100
                        st.success(f"Ham ({ham_prob:.2f}%)", icon="✅")

                with col2:
                    st.subheader("Key Word Analysis:")
                    spammy, hammy = get_important_words(pipeline, user_input)
                    st.markdown("**Top words indicating spam:**")
                    st.warning(" , ".join([f"`{word}`" for word, score in spammy]) or "None found in text")
                    st.markdown("**Top words indicating ham:**")
                    st.info(" , ".join([f"`{word}`" for word, score in hammy]) or "None found in text")
            else:
                st.warning("Please enter some text to analyze.", icon="⚠️")

    #Tab 2: Model Performance
    with tab2:
        st.header("Classifier Performance Metrics")
        st.write("The model was evaluated on a held-out test set of 1,115 samples.")
        
        col1, col2 = st.columns(2)
        with col1:
            try:
                st.image('confusion_matrix.png', caption='Confusion Matrix')
            except FileNotFoundError:
                st.error("confusion_matrix.png not found. Please run train_model.py.")
        with col2:
            st.subheader("Key Metrics")
            st.metric("Overall Accuracy", "98.39%")
            st.metric("Spam Precision", "98.55%")
            st.metric("Spam Recall", "89.40%")
            st.info("**Precision:** When it predicts spam, how often is it right? \n**Recall:** Of all actual spam, how many did it catch?")

    #Tab 3: Dataset Insights
    with tab3:
        st.header("What's in the Data?")
        st.write("These word clouds show the most frequent words in spam and ham messages from the training dataset.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Common Spam Words")
            try:
                st.image('spam_wordcloud.png', caption='Most Frequent Spam Words')
            except FileNotFoundError:
                st.error("spam_wordcloud.png not found. Please run train_model.py.")
        with col2:
            st.subheader("Common Ham Words")
            try:
                st.image('ham_wordcloud.png', caption='Most Frequent Ham (Non-Spam) Words')
            except FileNotFoundError:
                st.error("ham_wordcloud.png not found. Please run train_model.py.")
else:
    st.error("Failed to load the machine learning model. Please ensure 'train_model.py' has been run successfully.")