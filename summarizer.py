# -*- coding: utf-8 -*-

import networkx 
import spacy

import itertools as it
import math
import re

default_sents = 3
default_kp = 5

nlp_pipeline = spacy.load('en')

    
def summarize_page(url, sent_count = default_sents, kp_count = default_kp):
    import bs4
    import requests
    
    try:
        data = requests.get(url).text
        soup = bs4.BeautifulSoup(data, "html.parser")
        # Find the tag with most paragraph tags as direct children
        body = max(soup.find_all(), 
                   key=lambda tag: len(tag.find_all('p', recursive=False)))
                   
        paragraphs = map(lambda p: p.text, body('p'))
        text = ' '.join(paragraphs)
        return summarize(text, sent_count, kp_count)
    except Exception as e:
        return ("Something went wrong: {}".format(str(e)), [])
    

def summarize(text, sent_count = default_sents, kp_count = default_kp):
    """
    Produces a summary of a given text and also finds the keyphrases of the text
    if desired.
    
    Keyword arguments:
    sent_count -- summary size 
    kp_count -- number of keyphrases 
    
    If the keyword arguments are less than one, they will be considered as a
    ratio of the length of text or total number of candidate keywords. If they 
    are more than one, they will be considered as a fixed count.
    
    Returns a tuple containing the summary and the list of keyphrases.
    """
    summary = ""
    
    doc = nlp_pipeline(text)
    
    if sent_count > 0:
        summary = text_summary(doc, sent_count)
                          
    top_phrases = []
    
    if kp_count > 0:
        top_phrases = find_keyphrases(doc, kp_count)
                                 
    return (summary, top_phrases)
    

def text_summary(doc, sent_count):
    """
    Summarizes given text using TextRank algorithm.
    
    :param doc: a spacy.Doc object
    :param sent_count: number (/ratio) of sentences in the summary
    """
    sents = list(enumerate(doc.sents))
    sent_graph = networkx.Graph()
    sent_graph.add_nodes_from(idx for idx, sent in sents)
    
    for i in sent_graph.nodes_iter():
        for j in sent_graph.nodes_iter():
            if i != j and not sent_graph.has_edge(i,j):
                similarity = sent_similarity(sents[i][1], sents[j][1])
                if similarity != 0:
                    sent_graph.add_edge(i,j, weight=similarity)
                    
    sent_ranks = networkx.pagerank_numpy(sent_graph)
    
    if 0 < sent_count < 1:
        sent_count = round(sent_count * len(sent_ranks))
    sent_count = int(sent_count)
        
    top_indices = [idx for idx, rank in
                   sorted(sent_ranks.items(), key=lambda s: s[1], reverse=True)[:sent_count]]
                          
    # Return the key sentences in chronological order
    top_sents = map(lambda i: sents[i][1], sorted(top_indices))
    summary = ' '.join(sent.text for sent in top_sents)
    
    return summary
    
     
def find_keyphrases(doc, kp_count):
    """
    Finds keyphrases of given text using TextRank algorithm.
    
    :param doc: a spacy.Doc object
    :param kp_count: number (/ratio) of keyphrases
    """
    tokens = [normalise(tok) for tok in doc]
    candidates = [normalise(*token) for token in ngrams(doc, 1)]
    
    word_graph = networkx.Graph()
    word_graph.add_nodes_from(set(candidates))
    word_graph.add_edges_from(zip(candidates, candidates[1:]))
    
    kw_ranks = networkx.pagerank_numpy(word_graph)
    
    if 0 < kp_count < 1:
        kp_count = round(kp_count * len(kw_ranks))
    kp_count = int(kp_count)
        
    top_words = {word : rank for word, rank in kw_ranks.items()}
                        
    keywords = set(top_words.keys())
    phrases = {}
    
    for tok in tokens:
        if tok in keywords:
            kp_words = [tok]
            kp_words.extend(it.takewhile(lambda t: t in keywords, tokens))
            n = len(kp_words)
            avg_rank = sum(top_words[w] for w in kp_words) / n
            phrases[' '.join(kp_words)] = avg_rank
    
    top_phrases = [kp for kp, rank in 
                   sorted(phrases.items(), key=lambda w: w[1], reverse=True)[:kp_count]]
                   
    return top_phrases    
                          
                          
def ngrams(doc, n, filter_stopwords=True, good_tags={'NOUN', 'PROPN', 'ADJ'}):
    """
    Extracts a list of n-grams from a sequence of spacy.Tokens. Optionally 
    filters stopwords and parts-of-speech tags.
    
    :param doc: sequence of spacy.Tokens
    :param n: number of tokens in an n-gram
    :param filter_stopwords: flag for stopword filtering
    :param good_tags: set of wanted POS tags
    """
    ngrams = (doc[i:i+n] for i in range(len(doc) - n + 1))
    ngrams = (ngram for ngram in ngrams 
              if not any(w.is_space or w.is_punct for w in ngram))
    
    if filter_stopwords:
        ngrams = (ngram for ngram in ngrams
                  if not ngram[0].is_stop and not ngram[-1].is_stop)
    if good_tags:
        ngrams = (ngram for ngram in ngrams 
                  if all(word.pos_ in good_tags for word in ngram))
    
    for ngram in ngrams:
        yield ngram
        
        
def normalise(term):
    """
    Parses a token or span of tokens into a lemmatized string.
    Proper nouns are not lemmatized.
    
    :param term: spacy.Token or spacy.Span to be lemmatized
    """
    if isinstance(term, spacy.tokens.token.Token):
        return term.text if term.pos_ == 'PROPN' else term.lemma_
    elif isinstance(term, spacy.tokens.span.Span):
        return ' '.join(word.text if word.pos_ == 'PROPN' else word.lemma_
                        for word in term)
    else:
        msg = "Normalisation requires a Token or Span, not {}.".format(type(term))
        raise TypeError(msg)
 
 
def sent_similarity(sent1, sent2):
    """
    Calculates a similary measure between two sentences.
    
    :param sent1: a spacy.Span object
    :param sent2: a spacy.Span object
    """
    s1 = set(normalise(tok) for tok in sent1 if not tok.is_stop)
    s2 = set(normalise(tok) for tok in sent2 if not tok.is_stop)
    
    common_words = len(s1 & s2)
    normalizing_factor = math.log10(len(s1) * len(s2))
    
    if normalizing_factor == 0:
        return 0
    
    return common_words / normalizing_factor
    
    
usage = """
Usage: summarize.py [args] <URL> 

Supported arguments:
-s --sentences the number of sentences in the summary
-k --keyphrases the number of keyphrases

If the arguments are specifiec as decimal numbers smaller than one, they are 
considered as ratios with respect to the original text.
"""
    
if __name__ == "__main__":
    import argparse
    import sys
    
    if len(sys.argv) == 0:
        print(usage)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("-s", "--sentences", type=float, dest="sent_count", 
                        default=default_sents)
    parser.add_argument("-k", "--keyphrases", type=float, dest="kp_count", 
                        default=default_kp)
    args = parser.parse_args()
    
    res = summarize_page(args.url, args.sent_count, args.kp_count)
    print("{} \nKeyphrases: {}".format(res[0], res[1]))
    