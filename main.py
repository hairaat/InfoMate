import os
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from textblob import TextBlob
from transformers import pipeline

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load the pre-trained model and tokenizer
chatbot_pipeline = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# Define responses
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey!"],
    "goodbye": ["Goodbye!", "Bye!", "See you later!"],
    "thanks": ["You're welcome!", "No problem!", "Anytime!"],
    "default": ["Sorry, I didn't understand that.", "Could you please rephrase that?", "I'm not sure I follow."]
}

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define function to preprocess user input
def preprocess_input(text):
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# Define function for named entity recognition
def extract_entities(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    entities = ne_chunk(tagged_tokens)
    return entities

# Define function for sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Define function to generate response
def generate_response(input_text):
    input_tokens = preprocess_input(input_text)
    entities = extract_entities(input_text)
    sentiment_score = analyze_sentiment(input_text)

    if sentiment_score > 0.5:
        return "I'm glad you're feeling positive!"
    elif sentiment_score < -0.5:
        return "I'm sorry to hear that you're feeling negative."

    for entity in entities:
        if hasattr(entity, 'label'):
            if entity.label() == 'PERSON':
                return f"Tell me more about {entity[0][0]}."

    response = chatbot_pipeline(input_text, max_length=100)[0]['generated_text']
    return response.strip()

# Main loop
print("Chatbot: Hi! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'bye':
        print("Chatbot:", "Goodbye! Have a great day!")
        break
    else:
        print("Chatbot:", generate_response(user_input))
