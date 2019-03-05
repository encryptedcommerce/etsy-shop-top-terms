from nltk import download
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer

def update_nltk_resources(debug=True):
    quiet = not debug
    download('punkt', quiet=quiet)
    download('wordnet', quiet=quiet)
    download('stopwords', quiet=quiet)
    download('averaged_perceptron_tagger', quiet=quiet)

def provision_tokenizer():
    """Provides the Punkt word tokenizer.

    Returns:
        function: The Punk word tokenizer.
    """
    return word_tokenize

def provision_stemmer(stemmer='porter', debug=False):
    """Provides the specified stemmer.

    Returns:
        nltk.stem.porter.PorterStemmer or nltk.stem.lancaster.LancasterStemmer: Stemmer.
    """
    stemmer = stemmer.lower()
    if debug:
        print(f'Selecting {stemmer} as the stemming algorithm.')
    if stemmer == 'porter':
        return PorterStemmer()
    if stemmer == 'lancaster':
        return LancasterStemmer()
    raise ValueError('The specified stemmer is not supported.')

def provision_lemmatizer(lexical_db='wordnet'):
    """Provides a lemmatizer from the WordNet lexical database.

    Args:
        lexical_db (str): Lexical DB (currently only WordNet is supported).
    Returns:
        nltk.stem.wordnet.WordNetLemmatizer: WordNet lemmatizer.
    """
    lexical_db = lexical_db.lower()
    if lexical_db == 'wordnet':
        wnl = WordNetLemmatizer()
        return wnl
    raise ValueError('The specified lexical database is not supported.')

def apply_fn_to_tokenized_phrase(phrase, function):
    """Tokenizes a phrase and applies a transformation function to each word.
    
    Args: 
        phrase (str): The input phrase.
        function (function): The function used to transform each word.
    Returns:
        str: the transformed phrase.
    """
    token_words = word_tokenize(phrase)
    transformed_phrase=[]
    for word in token_words:
        transformed_phrase.append(function(word))
    return ' '.join(transformed_phrase)

