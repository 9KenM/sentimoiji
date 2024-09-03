from flask import Flask, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer
from textblob import TextBlob

nltk.download('vader_lexicon')

tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
nlp = pipeline('sentiment-analysis', model="finiteautomata/bertweet-base-sentiment-analysis")

app = Flask(__name__, static_url_path='')

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

def accurate_analyze_sentiment(text):
    maxChunkSize = 120
    tokens = tokenizer.tokenize(text)
    
    if len(tokens) <= maxChunkSize:
        result = nlp(text)[0]
        return result['label'], result['score']
    else:
        sentiments = []
        for i in range(0, len(tokens), maxChunkSize):
            chunk = tokens[i:i+maxChunkSize]
            chunk_text = tokenizer.convert_tokens_to_string(chunk)
            sentiment = nlp(chunk_text)[0]
            sentiments.append((sentiment['label'], sentiment['score']))
        
        # Calculate weighted average of sentiments
        total_score = sum(score for _, score in sentiments)
        weighted_sentiments = {
            'POS': sum(score for label, score in sentiments if label == 'POS') / total_score,
            'NEU': sum(score for label, score in sentiments if label == 'NEU') / total_score,
            'NEG': sum(score for label, score in sentiments if label == 'NEG') / total_score
        }
        
        sentiment = max(weighted_sentiments, key=weighted_sentiments.get)
        score = weighted_sentiments[sentiment]
        
        return sentiment, score

def map_sentiment_to_emotion(sentiment, score):
    if sentiment == "POS":
        if score > 0.9:
            return "Joy"
        elif score > 0.7:
            return "Excitement"
        elif score > 0.5:
            return "Optimism"
        else:
            return "Approval"
    elif sentiment == "NEG":
        if score > 0.9:
            return "Anger"
        elif score > 0.7:
            return "Sadness"
        elif score > 0.5:
            return "Disappointment"
        else:
            return "Annoyance"
    else:
        if score > 0.7:
            return "Confusion"
        else:
            return "Neutral"

@app.route('/accurate-sentiment', methods=['POST'])
def get_accurate_sentiment():
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({'error': 'Missing "text" parameter'}), 400
    
    text = data['text']
    sentiment, score = accurate_analyze_sentiment(text)
    emotion = map_sentiment_to_emotion(sentiment, score)
    
    emoji = {
        "Joy": "😂", "Excitement": "😆", "Optimism": "😃", "Approval": "👍",
        "Anger": "😡", "Sadness": "😔", "Disappointment": "😞", "Annoyance": "😠",
        "Confusion": "🤔", "Neutral": "😐"
    }[emotion]
    
    return jsonify({'text': text, 'sentiment': sentiment, 'emotion': emotion, 'score': score, 'emoji': emoji})

@app.route('/fast-sentiment', methods=['POST'])
def get_fast_sentiment():
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({'error': 'Missing "text" parameter'}), 400
    
    text = data['text']
    sentiment, emotion = fast_analyze_sentiment(text)
    
    emoji = {
        "Joy": "😂", "Excitement": "😆", "Optimism": "😃", "Approval": "👍",
        "Amusement": "😄", "Anger": "😡", "Grief": "😭", "Sadness": "😔",
        "Disappointment": "😞", "Annoyance": "😠", "Confusion": "🤔", "Neutral": "😐"
    }[emotion]
    
    return jsonify({'text': text, 'sentiment': sentiment, 'emotion': emotion, 'emoji': emoji})

@app.route('/')
def server_static_index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
