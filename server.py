from flask import Flask, request, jsonify
from sentiment_analysis.fast_analysis import fast_analyze_sentiment
from sentiment_analysis.accurate_analysis import accurate_analyze_sentiment, map_sentiment_to_emotion
from utils.emoji_mapper import get_emoji

app = Flask(__name__, static_url_path='')

@app.route('/accurate-sentiment', methods=['POST'])
def get_accurate_sentiment():
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({'error': 'Missing "text" parameter'}), 400
    
    text = data['text']
    sentiment, score = accurate_analyze_sentiment(text)
    emotion = map_sentiment_to_emotion(sentiment, score)
    emoji = get_emoji(emotion)
    
    return jsonify({'text': text, 'sentiment': sentiment, 'emotion': emotion, 'score': score, 'emoji': emoji})

@app.route('/fast-sentiment', methods=['POST'])
def get_fast_sentiment():
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({'error': 'Missing "text" parameter'}), 400
    
    text = data['text']
    sentiment, emotion = fast_analyze_sentiment(text)
    emoji = get_emoji(emotion)
    
    return jsonify({'text': text, 'sentiment': sentiment, 'emotion': emotion, 'emoji': emoji})

@app.route('/')
def server_static_index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
