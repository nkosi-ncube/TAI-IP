from textblob import TextBlob
import nltk
nltk.download('punkt')

# Take user input
user_input = input("Enter some text to analyze sentiment: ")

# Analyze sentiment
blob = TextBlob(user_input)
sentiment_score = blob.sentiment.polarity

# Define sentiment labels
if sentiment_score > 0:
    sentiment_label = "Positive"
elif sentiment_score < 0:
    sentiment_label = "Negative"
else:
    sentiment_label = "Neutral"

# Output sentiment
print("Sentiment Score:", sentiment_score)
print("Sentiment:", sentiment_label)
