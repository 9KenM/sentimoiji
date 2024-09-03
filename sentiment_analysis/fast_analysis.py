import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download('vader_lexicon')

def fast_analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    blob = TextBlob(text)
    
    # Determine sentiment and emotion
    if sentiment_scores['compound'] >= 0.5:
        sentiment = "Positive"
        if blob.sentiment.subjectivity > 0.7:
            emotion = "Excitement"
        elif sentiment_scores['pos'] > 0.8:
            emotion = "Joy"
        else:
            emotion = "Optimism"
    elif sentiment_scores['compound'] >= 0.05:
        sentiment = "Positive"
        if blob.sentiment.subjectivity > 0.6:
            emotion = "Amusement"
        else:
            emotion = "Approval"
    elif sentiment_scores['compound'] <= -0.5:
        sentiment = "Negative"
        if blob.sentiment.subjectivity > 0.7:
            emotion = "Anger"
        elif sentiment_scores['neg'] > 0.8:
            emotion = "Grief"
        else:
            emotion = "Sadness"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "Negative"
        if blob.sentiment.subjectivity > 0.6:
            emotion = "Annoyance"
        else:
            emotion = "Disappointment"
    else:
        sentiment = "Neutral"
        if blob.sentiment.subjectivity > 0.5:
            emotion = "Confusion"
        else:
            emotion = "Neutral"
    
    return sentiment, emotion
