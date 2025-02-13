import requests
import nltk

from nltk.corpus import words

# Download NLTK data
nltk.download('words')
nltk.download('punkt_tab')

word_list = set(words.words())
'''
from nltk.tokenize import sent_tokenize
test_sentence = "This is a test sentence. This is another sentence."
print(sent_tokenize(test_sentence))
'''
def escape_markdown(text):
    """Escape markdown special characters."""
    escape_chars = '_*[]()~`>#+-=|{}.!'
    return ''.join(f'\\{char}' if char in escape_chars else char for char in text)


def fetch_wikipedia_info(disease_name):
    """Fetch and format disease information from Wikipedia."""
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": disease_name,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "redirects": 1,
    }

    try:
        response = requests.get(URL, params=params)
        response.raise_for_status()
        data = response.json()
        page = next(iter(data['query']['pages'].values()))

        if "extract" in page:
            sentences = nltk.sent_tokenize(page["extract"])
            return "\n".join(f"• {sentence}" for sentence in sentences)
        else:
            return "No detailed information available."
    except Exception as e:
        return f"Error fetching data: {str(e)}"

def is_valid_input(input_text):
    """Check if input text is meaningful."""
    if not input_text or not isinstance(input_text, str):
        return False
    
    tokens = nltk.word_tokenize(input_text.lower())
    return sum(token in word_list for token in tokens) >= len(tokens) / 2
