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
    
    # Determine sentiment
    if sentiment_scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    # Determine emotion
    if sentiment_scores['pos'] > 0.5:
        if blob.sentiment.subjectivity > 0.7:
            emotion = "Excited"
        else:
            emotion = "Happy"
    elif sentiment_scores['neg'] > 0.5:
        if blob.sentiment.subjectivity > 0.7:
            emotion = "Angry"
        else:
            emotion = "Sad"
    elif sentiment_scores['neu'] > 0.5:
        emotion = "Calm"
    else:
        emotion = "Mixed"
    
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

@app.route('/accurate-sentiment', methods=['POST'])
def get_accurate_sentiment():
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({'error': 'Missing "text" parameter'}), 400
    
    text = data['text']
    sentiment, score = accurate_analyze_sentiment(text)
    
    emotion = "Neutral"
    if sentiment == "POS":
        emotion = "Happy" if score < 0.8 else "Excited"
    elif sentiment == "NEG":
        emotion = "Sad" if score < 0.8 else "Angry"
    
    emoji = {
        "Happy": "😊", "Excited": "😃",
        "Sad": "😔", "Angry": "😠",
        "Neutral": "😐"
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
        "Happy": "😊", "Excited": "😃",
        "Sad": "😔", "Angry": "😠",
        "Calm": "😌", "Mixed": "😕"
    }[emotion]
    
    return jsonify({'text': text, 'sentiment': sentiment, 'emotion': emotion, 'emoji': emoji})

@app.route('/')
def server_static_index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
