"""
This module contains functions to compute various statistics about a lyrics (list of lines).

The following NLTK packages need to be downloaded:
    - stopwords
    - averaged_perceptron_tagger
"""

from textstat import lexicon_count, polysyllabcount, monosyllabcount, dale_chall_readability_score
from nltk.tokenize import word_tokenize
from nltk import pos_tag, map_tag
from metaphone import doublemetaphone

def tokenize_sentence(sentence, lowercase=False, alpha_filter=False):
    """
    Tokenize the given sentence into a list of words.

    Parameters
    ----------
    sentence: str
        The sentence to be tokenized.
    lowercase: bool, optional
        Whether to lowercase the tokens. Default is False.
    alpha_filter: bool, optional
        Whether to filter out non-alphabetic tokens. Default is False.

    Returns
    -------
    list of str
        The list of tokens (words) in the sentence.
    """
    tokens = word_tokenize(sentence)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    if alpha_filter:
        tokens = [token for token in tokens if token.isalpha()]
    return tokens

def compute_avg_token_length(tokens):
    """
    Compute the average length of the given list of tokens.

    Parameters
    ----------
    tokens: list of str
        The list of tokens.

    Returns
    -------
    float
        The average length of the tokens. Returns 0 if the list is empty.
    """
    if tokens:
        token_lengths = [len(token) for token in tokens]
        avg_token_length = sum(token_lengths) / len(token_lengths)
    else:
        avg_token_length = 0
    return avg_token_length

def compute_ttr(tokens):
    """
    Compute the type-token ratio (TTR) of the given list of tokens.

    The TTR is defined as the number of unique tokens (types) in the list
    divided by the total number of tokens in the list.

    Parameters
    ----------
    tokens: list of str
        The list of tokens.

    Returns
    -------
    float
        The TTR of the tokens. Returns 0 if the list is empty.
    """
    if tokens:
        ttr = len(list(set(tokens)))/len(tokens)
    else:
        ttr = 0
    return ttr

def compute_avg_token_frequency(tokens, all_tokens):
    """
    Compute the average frequency of the given list of tokens in a larger list of tokens.

    Parameters
    ----------
    tokens: list of str
        The list of tokens for which the average frequency is to be computed.
    all_tokens: list of str
        The list of all tokens in which the frequency of the given tokens is to be measured.

    Returns
    -------
    float
        The average frequency of the tokens in the list of all tokens.
        Returns 0 if the list of tokens is empty.
    """
    if tokens:
        token_frequencies = [all_tokens.count(token) for token in tokens]
        avg_token_frequency = sum(token_frequencies) / len(token_frequencies)
    else:
        avg_token_frequency = 0
    return avg_token_frequency

def compute_alliteration_score(tokens):
    """
    Compute the alliteration score of the given list of tokens.
    It uses Metaphone phonetic algorithm.

    Parameters
    ----------
    tokens: list of str
        The list of tokens.

    Returns
    -------
    float
        The alliteration score for the given list of tokens.

    """
    alliteration_score = 0
    if tokens:
        # Get sequence of phonemes as a list
        phonemes = [meta_phoneme for token in tokens
            for meta_phoneme in doublemetaphone(token)[0]]
        if phonemes:
            unique_phonemes = list(set(phonemes))
            alliteration_score = (1 - len(unique_phonemes)/len(phonemes)) * len(phonemes)
    return alliteration_score

def count_uni_pos_tags(tagged_tokens):
    """
    Count the occurrences of the universal part-of-speech (POS) tags in a list of tagged tokens.

    Parameters
    ----------
    tagged_tokens: list of tuple
        The list of tagged tokens, where each tuple consists of a token and its POS tag.
        The POS tags must be in the Penn Treebank (PTB) tagset (NLTK POS tagging output).

    Returns
    -------
    dict
        A dictionary with the universal POS tags as keys and the counts as values.
    Notes
    -----
    The input tagged tokens are first mapped to the universal tagset of Petrov, Das, & McDonald.
    Interjections (tagged with UH in the PTB tagset) are kept distinct and mapped to INTJ.
    The resulting dictionary includes the counts of the following universal POS tags,
    including interjections:
        VERB, NOUN, PRON, ADJ, ADV, ADP, CONJ, DET, NUM, PRT, X, ., INTJ
    """
    # Map NLTK tokens in universal tagset of Petrov, Das, & McDonald
    tagged_tokens = [(token, map_tag('en-ptb', 'universal', tag))
        if tag != 'UH' else (token, 'INTJ')
        for token, tag in tagged_tokens]
    # Get sequence of POS tags as a list
    mapped_pos_tags = [tag for _, tag in tagged_tokens]
    uni_pos_tags = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP',
                    'CONJ', 'DET', 'NUM', 'PRT', 'X', '.', 'INTJ']
    uni_pos_counts = {}
    for uni_pos_tag in uni_pos_tags:
        uni_pos_counts[uni_pos_tag] = mapped_pos_tags.count(uni_pos_tag)
    return uni_pos_counts


def lyrics_statistics(lyrics):
    """
    Compute various statistics for a given lyrics.

    The statistics include:
    - verse length
    - count of monosyllabic words
    - count of polysyllabic words
    - average token length
    - readability score
    - normalized verse frequency
    - type-token ratio (TTR) of each verse
    - average frequency of each token in the verse (computed on the entire lyrics)
    - average frequency of each phoneme in the verse
    - count of verbs, nouns, adjectives, and adverbs in each verse

    Parameters
    ----------
    lyrics: list of str
        The lyrics, where each element is a verse.

    Returns
    -------
    dict
        A dictionary with the statistics as keys and lists of values as values.
        The lists of values correspond to the statistics of each verse in the lyrics.
    """
    # Initialize dict with all features
    stats = {
        'verse_length': [],
        'monosyl_words_count': [],
        'polysyl_words_count': [],
        'avg_token_length': [],
        'readability_score': [],
        'normalized_verse_frequency': [],
        'verse_ttr': [],
        'avg_token_frequency': [],
        'avg_phoneme_frequency': [],
        'VERB_count': [],
        'NOUN_count': [],
        'ADJ_count': [],
        'ADV_count': [],
        'INTJ_count': []
    }
    # Get all tokens before beginning iteration
    all_tokens = [token for verse in lyrics for token in tokenize_sentence(
        verse, lowercase=True, alpha_filter=True)]
    # Iterate trough verses
    for verse in lyrics:
        # Compute stats that require using textstat and not tokenization
        verse_length = lexicon_count(verse, removepunct=True)
        stats['verse_length'].append(verse_length)
        monosyl_words_count = polysyllabcount(verse)
        stats['monosyl_words_count'].append(monosyl_words_count)
        polysyl_words_count = monosyllabcount(verse)
        stats['polysyl_words_count'].append(polysyl_words_count)
        # Compute readability scores with textstats
        readability_score = dale_chall_readability_score(verse)
        stats['readability_score'].append(readability_score)
        normalized_verse_frequency = lyrics.count(verse)/len(lyrics)
        stats['normalized_verse_frequency'].append(normalized_verse_frequency)
        # Compute stats that require tokenization (NLTK)
        # Tokenize sentence doing lowercasing
        # and excluding punctuation and non alphabetic characters
        filtered_tokens = tokenize_sentence(
            verse, lowercase=True, alpha_filter=True)
        # Compute avg token length
        avg_token_length = compute_avg_token_length(filtered_tokens)
        stats['avg_token_length'].append(avg_token_length)
        # Compute verse TTR of the verse (richness)
        verse_ttr = compute_ttr(filtered_tokens)
        stats['verse_ttr'].append(verse_ttr)
        # Compute avg token frequency
        avg_token_frequency = compute_avg_token_frequency(
            filtered_tokens, all_tokens)
        stats['avg_token_frequency'].append(avg_token_frequency)
        # Compute avg phoneme frequency
        avg_phoneme_frequency = compute_alliteration_score(filtered_tokens)
        stats['avg_phoneme_frequency'].append(avg_phoneme_frequency)

        # Count parts-of-speech
        tagged_tokens = pos_tag(tokenize_sentence(verse))
        pos_counts = count_uni_pos_tags(tagged_tokens)
        stats['VERB_count'].append(pos_counts['VERB'])
        stats['NOUN_count'].append(pos_counts['NOUN'])
        stats['ADJ_count'].append(pos_counts['ADJ'])
        stats['ADV_count'].append(pos_counts['ADV'])
        stats['INTJ_count'].append(pos_counts['INTJ'])
    return stats


fake_lyrics = [    
    "Unstoppable, yeah, unstoppable, yeah, ah",
    "I wake up every morning, with the sun in my eyes",
    "I stumble out of bed, and I hit the ground running",
    "I've got a lot on my plate, but I don't mind the load",
    "I know I've got what it takes, to make it down the road",
    "I'm living for today, and I'm chasing my dreams",
    "I won't let anyone stand in my way, or burst at the seams",
    "I'm on a mission, to reach for the stars",
    "I'm unstoppable, behind the steering wheel of my car",
    "I hit the pavement, and I don't look back",
    "I've got my foot on the gas, and I'm on the right track",
    "I know I've got what it takes, to make it to the top",
    "I won't stop until I reach the mountaintop",
    "I'm unstoppable, behind the steering wheel of my car",
    "I'm on a mission, to reach for the stars",
    "I won't let anyone stand in my way, or burst at the seams",
    "I'm living for today, and I'm chasing my dreams",
    "I've got my eyes on the prize, and I won't let it go",
    "I'll keep on fighting, with all my might and my mojo"
    "I'll keep on fighting, with all my might and my mojo",
]

if __name__ == '__main__':
    print(lyrics_statistics(fake_lyrics))
