# -*- coding: utf-8 -*-

from nltk.corpus import stopwords

import networkx 
import nltk

import itertools as it
import math
import re

default_sents = 3
default_kp = 5

# Tokenization regexp from the NLTK Book
pattern = r"""(?x)
              (?:[A-Z]\.)+       # abbreviations, e.g. U.S.A.
              |\d+(?:\.\d+)?%?   # numbers, incl. currency and percentages
              |\w+(?:[-']\w+)*   # words w/ optional internal hyphens/apostrophe
              |(?:[+/\-@&*])     # special characters with meanings
"""

stopwords = stopwords.words('english')
tokenizer = nltk.tokenize.RegexpTokenizer(pattern)
wnl = nltk.WordNetLemmatizer()


def normalise(word):
    word = word.lower()
    word = wnl.lemmatize(word)
    return word
    
    
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
    sent_count -- summary size (default 0.2)
    kp_count -- number of keyphrases (default 5)
    
    If the keyword arguments are less than one, they will be considered as a
    ratio of the length of text or total number of candidate keywords. If they 
    are more than one, they will be considered as a fixed count.
    
    Returns a tuple containing the summary and the list of keyphrases.
    """
    summary = ""
    
    sents = [(idx, sent) for idx, sent in enumerate(nltk.sent_tokenize(text))]
    words = [[normalise(word) for word in tokenizer.tokenize(sent)] 
             for (idx, sent) in sents]
        
    
    if sent_count > 0:
        summary = text_summary(sents, words, sent_count)
                          
    top_phrases = []
    
    if kp_count > 0:
        words = list(it.chain.from_iterable(words))
        top_phrases = find_keyphrases(words, kp_count)
                                 
    return (summary, top_phrases)
    

def text_summary(sents, words, sent_count):
    """
    Summarizes given text using TextRank algorithm.
    
    :param sents: iterable of the sentences of the text
    :param words: list of lists containing the normalised words of each sentence
    :param sent_count: number (/ratio) of sentences in the summary
    """
    sent_graph = networkx.Graph()
    sent_graph.add_nodes_from(idx for idx, sent in sents)
    
    for i in sent_graph.nodes_iter():
        for j in sent_graph.nodes_iter():
            if i != j and not sent_graph.has_edge(i,j):
                similarity = sent_similarity(words[i], words[j])
                if similarity != 0:
                    sent_graph.add_edge(i,j, weight=similarity)
                    
    sent_ranks = networkx.pagerank(sent_graph)
    
    if 0 < sent_count < 1:
        sent_count = round(sent_count * len(sent_ranks))
    sent_count = int(sent_count)
        
    top_indices = [idx for idx, rank in
                   sorted(sent_ranks.items(), key=lambda s: s[1], 
                          reverse=True)[:sent_count]]
                          
    # Return the key sentences in chronological order
    top_sents = map(lambda i: sents[i][1], sorted(top_indices))
    summary = ' '.join(top_sents)
    
    return summary
    
    
def candidate_keywords(words):
    # Select adjectives and nouns
    good_tags = set(['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS'])
    pos_tokens = nltk.pos_tag(words)
    candidates = [word for word, tag in pos_tokens 
                  if tag in good_tags and word not in stopwords]
    return candidates
    
     
def find_keyphrases(words, kp_count):
    """
    Finds keyphrases of given text using TextRank algorithm.
    
    :param words: list of normalised words in the text
    :param kp_count: number (/ratio) of keyphrases
    """
    word_graph = networkx.Graph()
    candidates = candidate_keywords(words)
    word_graph.add_nodes_from(set(candidates))
    add_edges(candidates, word_graph)
    
    kw_ranks = networkx.pagerank(word_graph)
    
    if 0 < kp_count < 1:
        kp_count = round(kp_count * len(kw_ranks))
    kp_count = int(kp_count)
        
    top_words = {word : rank
                 for word, rank in 
                 sorted(kw_ranks.items(),
                        key=lambda w: w[1],
                        reverse=True)}
                        
    keywords = set(top_words.keys())
    phrases = {}
    
    word_iter = iter(words)
    for word in word_iter:
        if word in keywords:
            kp_words = [word]
            kp_words.extend(it.takewhile(lambda w: w in keywords, word_iter))
            n = len(kp_words)
            avg_rank = sum(top_words[w] for w in kp_words) / n
            phrases[' '.join(kp_words)] = avg_rank
    
    top_phrases = [kp for kp, rank in 
                   sorted(phrases.items(), 
                          key=lambda w: w[1], 
                          reverse=True)[:kp_count]]
                   
    return top_phrases    
                          
                          
def add_edges(word_list, graph):
    """
    Adds edges to the keyword graph. This function considers bigrams, i.e.
    collocated words.
    """
    graph.add_edges_from( zip(word_list, word_list[1:]) )
 
 
def sent_similarity(word_list1, word_list2):
    s1 = set(w for w in word_list1 if w not in stopwords)
    s2 = set(w for w in word_list1 if w not in stopwords)
    
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
    print("{} \nKeyphrases: {}".format(res[0].encode("utf-8"), res[1]))
    