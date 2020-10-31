import numpy
import nltk
import json
import pickle
import numpy as np
import random
import os
import spacy
import pandas as pd
import os.path
import openai
from color import green_string, blue_string
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk import ngrams, FreqDist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
from nltk.corpus import stopwords

# Attempts to read a pickle file
# If no pickle file exists, it will create one with the content "default"


def attempt_to_read_pickle(name, default):
    try:
        new_pickle = pickle.load(open(name, "rb"))
    except (OSError, IOError) as e:
        new_pickle = default
        pickle.dump(new_pickle, open(name, "wb"))

    return new_pickle

# Creates a new user dictionary
# Had to use a dictionary because pickle doesn't like classes


def generate_new_user():
    return {
        "name": "",
        "likes": [],
        "dislikes": []
    }


# Helps limit the amount of output that tensorflow does when training and running models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# loads up the small model for spacy
nlp = spacy.load("en_core_web_sm")

# Creates a lemmatizer
lemmatizer = WordNetLemmatizer()

# Create a user object for writing to
user = attempt_to_read_pickle("user.pickle", generate_new_user())

# These are used for training the model but won't be needed for you to run it
# force_training = False
# should_train_model = (not os.path.isfile("chatbot_model.h5") or force_training)

# if(should_train_model):
#     train()

# Loads the tensorflow model from the h5 file
model = load_model('chatbot_model.h5')

# Loads up the intents, words, and classes used to train model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pickle', 'rb'))
classes = pickle.load(open('classes.pickle', 'rb'))


# Just a helper function to let the AI speak
def ai_say(text):
    print(green_string("AI: ") + text)


# Create a bag of words vector from a sentence and the existing words from the pickle file
def bag_of_words(sentence):
    tokenized_words = nltk.word_tokenize(sentence)
    lemmatized_and_lowercased = [lemmatizer.lemmatize(
        word.lower()) for word in tokenized_words]
    return(np.array([1 if (word in lemmatized_and_lowercased) else 0 for word in words]))


# Uses the tensorflow model to predict which class a given sentence belongs to
def predict(sentence, model):
    bag = bag_of_words(sentence)
    prediction = model.predict(np.array([bag]))[0]
    return classes[numpy.argmax(prediction)]


# Performs named entity recognition on a sentence to find the name you are trying to supply
def perform_ner(text):
    ne_tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))

    for chunk in ne_tree:
        if hasattr(chunk, "label") and chunk.label() == "PERSON":
            # If it finds a name then it stored it in the user object
            name = ' '.join(c[0] for c in chunk)
            ai_say("It's great to meet you, " + name + ".")
            user["name"] = name
            return name

    # If it doesn't find a name then it asks for more clarification
    ai_say("Were you trying to tell me your name? I didn't quite catch it.")
    return ""

# This is the response the the "What have you learned" query


def respond_to_information_query(text):
    # Print out the users name they've supplied it
    if(user["name"] != ""):
        ai_say("I know your name is " + user["name"] + ".")
    else:
        ai_say("I don't know your name yet. Could you tell me, please?")

    # Print out the users likes if they've supplied it
    if(len(user["likes"])):
        ai_say("I've learned that you really like " + ", ".join(user["likes"]))
    else:
        ai_say("I haven't learned much about what you like. Would you mind telling me something you love?")

    # Print out the users dislikes if they've supplied them
    if(len(user["dislikes"])):
        ai_say("I've learned that you really dislike " +
               ", ".join(user["dislikes"]))
    else:
        ai_say("I haven't learned much about what you dislike. Would you mind telling me something you hate?")

    # Print out the most common terms that the user has said
    common_terms = get_common_terms(
        [token for token in nltk.tokenize.word_tokenize(" ".join(all_chat_messages))])
    ai_say("Some of your most 2 word common phrases are: " + ", ".join(
        [frequency_tuple[0][0] + " " + frequency_tuple[0][1] for frequency_tuple in common_terms[2].most_common(5)]))
    ai_say("Some of your most 3 word common phrases are: " + ", ".join(
        [frequency_tuple[0][0] + " " + frequency_tuple[0][1] + " " + frequency_tuple[0][2] for frequency_tuple in common_terms[3].most_common(5)]))


# Gets a frequency distirbution of the most common ngrams from the chat log
def get_common_terms(all_messages):
    all_counts = dict()
    for size in 2, 3, 4, 5:
        all_counts[size] = FreqDist(ngrams(all_messages, size))

    return all_counts


# Checks to see if a users sentiment score is highly positive or negative
def check_for_likes_and_dislikes(text):
    testimonial = TextBlob(text)

    positive_threshold = 0.5
    negative_threshold = -positive_threshold

    # If it is highly positive or negative, then we are going to add smoething to their likes or dislikes
    if testimonial.polarity >= positive_threshold:
        ai_say("You liked something a lot in there!")
    elif testimonial.polarity <= negative_threshold:
        ai_say("You disliked something a lot in there!")
    else:
        return

    # Try to find the subject of the sentence through NER, or through tagging with SpaCy
    name = ""
    ne_tree = nltk.ne_chunk(nltk.pos_tag(
        nltk.word_tokenize(text)), binary=True)

    for chunk in ne_tree:
        if hasattr(chunk, "label") and chunk.label() == "NE":
            name = ' '.join(c[0] for c in chunk)

    doc = nlp(text)
    direct_objects = [tok for tok in doc if (tok.dep_ == "dobj")]
    noun_subjects = [tok for tok in doc if (tok.dep_ == "nsubj")]

    emotional_object = name if name else direct_objects[0] if len(
        direct_objects) and direct_objects[0] else noun_subjects[0] if len(noun_subjects) and noun_subjects[0] else ""

    # Print out the thing that they liked or disliked
    if emotional_object and testimonial.polarity >= positive_threshold:
        ai_say("I think I've detected that you like \"" +
               str(emotional_object) + "\"")
        user["likes"].append(str(emotional_object))
    elif emotional_object and testimonial.polarity <= negative_threshold:
        ai_say("I think I've detected that you dislike \"" +
               str(emotional_object) + "\"")
        user["dislikes"].append(str(emotional_object))
    else:
        ai_say("I detected a lot of emotion in your message. If you're trying to tell me you like or dislike something, try simplifying your language.")


# If the bot has no response it will use some of these responses
misunderstood = ["I don't understand.", "Could you rephrase please?",
                 "I'm not sure I'm catching on to what you're asking."]


# These responses are static
baked_responses = ["hello", "functionality", "whatis", "whoplays", "howwork", "platform",
                   "who", "hate", "modes", "rumble", "dropshot", "hoops", "snowday", "esports", "howmany", "purchase"]

# These responses are not static and run some code when they are triggered
non_baked_responses = [
    {
        "class": "information",
        "code": respond_to_information_query
    },
    {
        "class": "name-statement",
        "code": perform_ner
    }
]

# The initial introductory sequence that tells you what the bot does
print(f"{green_string('AI: Hello!')} I am a ChatBot created for {green_string('Human Language Technologies')}. Ask me anything about the video game {green_string('Rocket League!')}")
ai_say("You can ask what Rocket League is, how the game is played, who plays it professionally, which platforms support Rocket League, etc.")

# Creates the initial pickle file for saving chat messages
all_chat_messages = attempt_to_read_pickle("chatlog.pickle", [])

# Starts off the conversation
chat_input = input(blue_string("You: "))

# A way to exit out of the main loop, main converstaion loop
while chat_input.lower() != "quit":
    # Add your message to the message history and save
    all_chat_messages.append(chat_input)
    pickle.dump(all_chat_messages, open("chatlog.pickle", "wb"))

    # Predict what kind of question you are asking the model using tensorflow
    predicted_class = predict(chat_input, model)

    # If the predicted class is a static response then give a static response
    if predicted_class in baked_responses:
        responses = [intent['responses']
                     for intent in intents['intents'] if intent['class'] == predicted_class]

        if(len(responses) > 0):
            ai_say(random.choice(responses[0]))
        else:
            ai_say(random.choice(misunderstood))

    # If the predicted class is not a static response, run the code responsible for handling this response
    # This is a functional programming technique
    else:
        [response for response in non_baked_responses if response['class']
            == predicted_class][0]["code"](chat_input)

    # Check to see if there are high sentiment scores on the message
    check_for_likes_and_dislikes(chat_input)

    # Resave the user pickle file in case name changes, like or dislikes changes
    pickle.dump(user, open("user.pickle", "wb"))

    # Finally prompt for more input
    chat_input = input(blue_string("You: "))
