import os

# Suppress TensorFlow warnings about logs that are less important (to avoid clutter)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import pickle  # For saving/loading Python objects (like words, classes)
import json  # For handling JSON data, if needed elsewhere in the code
import numpy as np  # For numerical operations and arrays
import nltk  # Natural Language Toolkit for text processing
from nltk.stem import WordNetLemmatizer  # For converting words to their base form (lemmatization)

from tensorflow.keras.models import load_model  # For loading the trained Keras model

# A list of symbols to be ignored when processing text
ignore_symbols = ['?', '!', '.', ',']

# Correct directory for the model
model_dir = os.path.join(os.getcwd(), 'Flask Application', 'model')

# Function to clean up and preprocess a sentence
def clean_up_sentence(sentence):
    """
    This function tokenizes the input sentence and lemmatizes the words.
    It also removes any symbols that are in the 'ignore_symbols' list.
    
    Args:
    sentence (str): The input sentence that needs to be cleaned.
    
    Returns:
    list: A list of lemmatized words from the sentence.
    """
    
    # Create a lemmatizer object to reduce words to their base form
    lemmatizer = WordNetLemmatizer()

    # Tokenize the sentence into words
    sentence_words = nltk.word_tokenize(sentence)
    
    # Lemmatize each word and ignore symbols in 'ignore_symbols'
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words if word not in ignore_symbols]

    return sentence_words


# Function to create a bag-of-words representation of the sentence
def bag_of_words(sentence):
    """
    Converts a given sentence into a bag-of-words representation,
    which is a binary vector indicating which words from the model's vocabulary are present in the sentence.
    
    Args:
    sentence (str): The input sentence.
    
    Returns:
    np.array: A bag-of-words array corresponding to the sentence.
    """
    
    # Load the vocabulary (words) that was saved during model training
    words = pickle.load(open(os.path.join(model_dir, 'words.pkl'), 'rb'))

    # Clean up the input sentence
    sentence_words = clean_up_sentence(sentence)
    
    # Initialize a bag (list) with zeros for each word in the vocabulary
    bag = [0] * len(words)

    # Set the corresponding index to 1 if the word exists in the sentence
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1 
    
    return np.array(bag)


# Function to predict the intent class of a sentence
def predict_class(sentence):
    """
    Predicts the intent of a given sentence using a pre-trained neural network model.
    
    Args:
    sentence (str): The input sentence for which intent is to be predicted.
    
    Returns:
    list: A list of intents and their respective probabilities, sorted in descending order of probability.
    """

    # Load the classes.pkl file from the correct directory
    classes = pickle.load(open(os.path.join(model_dir, 'classes.pkl'), 'rb'))
    model = load_model(os.path.join(model_dir, 'chatbot_model.keras'))

    # Convert the input sentence into a bag-of-words representation
    bow = bag_of_words(sentence)
    
    # Predict the intent probabilities for the sentence using the model
    res = model.predict(np.array([bow]))[0]

    # Set an error threshold to filter out predictions with low confidence
    ERROR_THRESHOLD = 0.25

    # Filter results that are above the threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Sort the results based on probability in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # Prepare a list to return intents with their probabilities
    return_list = []

    for r in results:
        # Append intent name and its probability to the return list
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    
    return return_list


def get_response(intents_list):
    """
    This function takes the predicted intents list (from the 'predict_class' function) and returns 
    an appropriate response based on the highest-ranked intent.
    
    Args:
    intents_list (list): A list of predicted intents with probabilities, 
                         typically output from the 'predict_class' function.
    
    Returns:
    str: A randomly selected response corresponding to the predicted intent.
    """
    model_dir = os.path.join(os.getcwd(), 'Flask Application')

    # Load the intents JSON file that contains all the predefined intents and their responses
    # Load the intents.json file with UTF-8 encoding
    with open(os.path.join(model_dir, 'intents.json'), 'r', encoding='utf-8') as f:
        intents_json = json.load(f)

    # Get the predicted intent tag with the highest probability
    tag = intents_list[0]['intent']
    
    # Get the list of all intents from the intents JSON
    list_of_intents = intents_json['intents']

    # Search for the intent that matches the predicted tag and select a random response from its list
    for i in list_of_intents:
        if i['tag'] == tag:  # Check if the current intent's tag matches the predicted tag
            result = random.choice(i['responses'])  # Pick a random response for this intent
            break  # Exit the loop once a matching tag is found
    
    # Return the selected response
    return result

