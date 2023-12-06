from flask import Flask, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer

nltk.download('vader_lexicon')

tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

nlp = pipeline('sentiment-analysis', model="finiteautomata/bertweet-base-sentiment-analysis")

app = Flask(__name__, static_url_path='')

def fast_analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)

    if sentiment_scores['compound'] >= 0.05:
        return "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def accurate_analyze_sentiment(text):
    maxChunkSize = 120 # break tokens into chunks of size 120
    tokens = tokenizer.tokenize(text)

    if len(tokens) <= maxChunkSize:
        result = nlp(text)[0]
        return result['label']
    else:
        sentiments = []
        for i in range(0, len(tokens), maxChunkSize):
            chunk = tokens[i:i+maxChunkSize]
            chunk_text = tokenizer.convert_tokens_to_string(chunk)
            sentiment = nlp(chunk_text)[0]
            sentiments.append(sentiment['label'])
        most_common_sentiment = max(set(sentiments), key=sentiments.count)
        return most_common_sentiment

@app.route('/accurate-sentiment', methods=['POST'])
def get_accurate_sentiment():
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({'error': 'Missing "text" parameter'}), 400
    
    text = data['text']
    result = accurate_analyze_sentiment(text)
    return jsonify({'text': text, 'sentiment': result})
    

@app.route('/fast-sentiment', methods=['POST'])
def get_fast_sentiment():
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({'error': 'Missing "text" parameter'}), 400
    
    text = data['text']
    result = fast_analyze_sentiment(text)
    
    return jsonify({'text': text, 'sentiment': result})

@app.route('/')
def server_static_index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)
