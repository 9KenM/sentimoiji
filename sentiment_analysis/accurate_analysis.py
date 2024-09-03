from transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
nlp = pipeline('sentiment-analysis', model="finiteautomata/bertweet-base-sentiment-analysis")

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
