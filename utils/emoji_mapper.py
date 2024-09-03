EMOJI_MAP = {
    "Joy": "😂", "Excitement": "😆", "Optimism": "😃", "Approval": "👍",
    "Amusement": "😄", "Anger": "😡", "Grief": "😭", "Sadness": "😔",
    "Disappointment": "😞", "Annoyance": "😠", "Confusion": "🤔", "Neutral": "😐"
}

def get_emoji(emotion):
    return EMOJI_MAP.get(emotion, "😐")
