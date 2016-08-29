# -*- coding: utf-8 -*-

"""
Unsupervised text keyphrase extraction and summarization utility.
Rasmus Heikkila, 2016
"""

from collections import Counter, defaultdict

import networkx
import spacy

import itertools as it
import math

default_sents = 3
default_kp = 5

nlp_pipeline = spacy.load('en')


def summarize_page(url, sent_count=default_sents, kp_count=default_kp):
    """
    Retrieves a web page, finds its body of content and summarizes it.

    Args:
        url: the url of the website to summarize
        sent_count: number(/ratio) of sentences in the summary
        kp_count: number(/ratio) of keyphrases in the summary
    Returns:
        A tuple (summary, keyphrases). Any exception will be returned
        as a tuple (message, []).
    """
    import bs4
    import requests

    try:
        data = requests.get(url).text
        soup = bs4.BeautifulSoup(data, "html.parser")
        # Find the tag with most paragraph tags as direct children
        body = max(soup.find_all(),
                   key=lambda tag: len(tag.find_all('p', recursive=False)))

        paragraphs = map(lambda p: p.text, body('p'))
        text = '\n'.join(paragraphs)
        return summarize(text, sent_count, kp_count)
    except Exception as e:
        return ("Something went wrong: {}".format(str(e)), [])


def summarize(text, sent_count=default_sents, kp_count=default_kp, idf=None):
    """
    Produces a summary of a given text and also finds the keyphrases of the text
    if desired.
    
    Args:
        text: the text string to summarize
        sent_count: number of sentences in the summary
        kp_count: number of keyphrases in the summary
        idf: a dictionary (string, float) of inverse document frequencies
    Returns:
        A tuple (summary, keyphrases).

    If sent_count and kp_count are less than one, they will be considered as a
    ratio of the length of text or total number of candidate keywords. If they
    are more than one, they will be considered as a fixed count.
    """
    summary = ""

    doc = nlp_pipeline(text)

    if sent_count > 0:
        summary = text_summary(doc, sent_count)

    top_phrases = []

    if kp_count > 0:
        top_phrases = sgrank(doc, kp_count, idf=idf)

    return (summary, top_phrases)


def text_summary(doc, sent_count):
    """
    Summarizes given text using TextRank algorithm.

    Args:
        doc: a spacy.Doc object
        sent_count: number (/ratio) of sentences in the summary
    Returns:
        Text summary
    """
    sents = list(enumerate(doc.sents))
    sent_graph = networkx.Graph()
    sent_graph.add_nodes_from(idx for idx, sent in sents)

    for i in sent_graph.nodes_iter():
        for j in sent_graph.nodes_iter():
            if i != j and not sent_graph.has_edge(i, j):
                similarity = sent_similarity(sents[i][1], sents[j][1])
                if similarity != 0:
                    sent_graph.add_edge(i, j, weight=similarity)

    sent_ranks = networkx.pagerank_scipy(sent_graph)

    if 0 < sent_count < 1:
        sent_count = round(sent_count * len(sent_ranks))
    sent_count = int(sent_count)

    top_indices = top_keys(sent_count, sent_ranks)

    # Return the key sentences in chronological order
    top_sents = map(lambda i: sents[i][1], sorted(top_indices))

    return format_output(doc, list(top_sents))


def sent_similarity(sent1, sent2):
    """
    Calculates a similary measure between two sentences.

    Args:
        sent1: a spacy.Span object
        sent2: a spacy.Span object
    """
    s1 = set(normalize(tok) for tok in sent1 if not tok.is_stop)
    s2 = set(normalize(tok) for tok in sent2 if not tok.is_stop)

    common_words = len(s1 & s2)
    normalizing_factor = math.log10(len(s1) * len(s2))

    if normalizing_factor == 0:
        return 0

    return common_words / normalizing_factor


def format_output(doc, sents):
    """
    Breaks the summarized text into paragraphs.

    Args:
        doc: a spacy.Doc object
        sents: a list of spacy.Spans, the sentences in the summary
    Returns:
        Text summary as a string with newlines
    """
    sent_iter = iter(sents)
    output = [next(sent_iter)]
    par_breaks = (idx for idx, tok in enumerate(doc) if '\n' in tok.text)

    try:
        # Find the first newline after first sentence
        idx = next(i for i in par_breaks if i >= output[0].end)
        for sent in sent_iter:
            if '\n' not in output[-1].text:
                if idx < sent.start:
                    # If there was no newline in the previous sentence
                    # and there is one in the text between the two sentences, add it
                    output.append(doc[idx])
            output.append(sent)
            idx = next(i for i in par_breaks if i >= sent.end)
    except StopIteration:
        # Add the rest of sentences if there are no more newlines
        output.extend(sent_iter)

    return ''.join(elem.text_with_ws for elem in output)


def sgrank(doc, kp_count, window=1500, idf=None):
    """
    Extracts keyphrases from a text using SGRank algorithm.

    Args:
        doc: a spacy.Doc object
        kp_count: number of keyphrases
        window: word co-occurrence window length
        idf: a dictionary (string, float) of inverse document frequencies
    Returns:
        list of keyphrases
    Raises:
        TypeError if idf is not dictionary or None
    """
    if isinstance(idf, dict):
        idf = defaultdict(lambda: 1, idf)
    elif idf is not None:
        msg = "idf must be a dictionary, not {}".format(type(idf))
        raise TypeError(msg)

    cutoff_factor = 3000
    token_count = len(doc)
    top_n = max(int(token_count * 0.2), 100)
    min_freq = 1

    if 1500 < token_count < 4000:
        min_freq = 2
    elif token_count >= 4000:
        min_freq = 3

    terms = [tok for toks in (ngrams(doc, n) for n in range(1,7)) for tok in toks]
    term_strs = {id(term): normalize(term) for term in terms}

    # Count terms and filter by the minimum term frequency
    term_freqs = {term_str: freq for term_str, freq in
                  Counter(term_strs[id(term)] for term in terms).items()
                  if freq >= min_freq}

    if idf:
        # For ngrams with n >= 2 we have idf = 1
        modified_tfidf = {term_str: freq * idf[term_str] if ' ' not in term_str else freq
                     for term_str, freq in term_freqs.items()}
        top_terms = set(top_keys(top_n, modified_tfidf))
    else:
        top_terms = set(top_keys(top_n, term_freqs))

    terms = [term for term in terms if term_strs[id(term)] in top_terms]
    term_weights = {}

    # Calculate term weights 
    for term in terms:
        term_str = term_strs[id(term)]
        term_len = math.sqrt(len(term))
        term_freq = term_freqs[term_str]
        first_occ = math.log(cutoff_factor / (term.start + 1))
        subsum_count = 0
        if term_len > 1:
            # Sum the frequencies of all other terms that contain this term
            subsum_count = sum(term_freqs[other] for other in top_terms
                               if other is not term_str and term_str in other)
        freq_diff = term_freq - subsum_count
        if idf and term_len == 1:
            freq_diff *= idf[term_str]
        term_weights[term_str] = freq_diff * first_occ * term_len

    # Use only positive-weighted terms
    terms = [term for term in terms if term_weights[term_strs[id(term)]] > 0]

    num_co_occurrences = defaultdict(lambda: defaultdict(int))
    total_log_distance = defaultdict(lambda: defaultdict(float))

    # Calculate term co-occurrences and co-occurrence distances within the co-occurrence window
    for t1, t2 in it.combinations(terms, 2):
        dist = abs(t1.start - t2.start)
        if dist <= window:
            t1_str = term_strs[id(t1)]
            t2_str = term_strs[id(t2)]
            if t1_str is not t2_str:
                num_co_occurrences[t1_str][t2_str] += 1
                total_log_distance[t1_str][t2_str] += math.log(window / max(1, dist))

    edge_weights = defaultdict(lambda: defaultdict(float))
    for t1, terms in total_log_distance.items():
        for t2 in terms:
            edge_weights[t1][t2] = (total_log_distance[t1][t2] / num_co_occurrences[t1][t2]) \
                                   * term_weights[t1] * term_weights[t2]

    # Normalize edge weights by sum of outgoing edge weights
    norm_edge_weights = []
    for t1, terms in edge_weights.items():
        weights_sum = sum(terms.values())
        norm_edge_weights.extend((t1, t2, weight / weights_sum)
                                 for t2, weight in terms.items())

    term_graph = networkx.DiGraph()
    term_graph.add_weighted_edges_from(norm_edge_weights)
    term_ranks = networkx.pagerank_scipy(term_graph)

    if 0 < kp_count < 1:
        kp_count = round(kp_count * len(term_ranks))
    kp_count = int(kp_count)

    top_phrases = top_keys(kp_count, term_ranks)

    return top_phrases


def textrank(doc, kp_count):
    """
    Extracts keyphrases of a text using TextRank algorithm.

    Args:
        doc: a spacy.Doc object
        kp_count: number of keyphrases
    Returns:
        list of keyphrases
    """
    tokens = [normalize(tok) for tok in doc]
    candidates = [normalize(*token) for token in ngrams(doc, 1)]

    word_graph = networkx.Graph()
    word_graph.add_nodes_from(set(candidates))
    word_graph.add_edges_from(zip(candidates, candidates[1:]))

    kw_ranks = networkx.pagerank_scipy(word_graph)

    if 0 < kp_count < 1:
        kp_count = round(kp_count * len(kw_ranks))
    kp_count = int(kp_count)

    top_words = {word: rank for word, rank in kw_ranks.items()}

    keywords = set(top_words.keys())
    phrases = {}

    tok_iter = iter(tokens)
    for tok in tok_iter:
        if tok in keywords:
            kp_words = [tok]
            kp_words.extend(it.takewhile(lambda t: t in keywords, tok_iter))
            n = len(kp_words)
            avg_rank = sum(top_words[w] for w in kp_words) / n
            phrases[' '.join(kp_words)] = avg_rank

    top_phrases = top_keys(kp_count, phrases)

    return top_phrases


def ngrams(doc, n, filter_stopwords=True, good_tags={'NOUN', 'PROPN', 'ADJ'}):
    """
    Extracts a list of n-grams from a sequence of tokens. Optionally
    filters stopwords and parts-of-speech tags.

    Args:
        doc: sequence of spacy.Tokens (for example: spacy.Doc)
        n: number of tokens in an n-gram
        filter_stopwords: flag for stopword filtering
        good_tags: set of accepted part-of-speech tags
    Returns:
         a generator of spacy.Spans
    """
    ngrams_ = (doc[i:i + n] for i in range(len(doc) - n + 1))
    ngrams_ = (ngram for ngram in ngrams_
               if not any(w.is_space or w.is_punct for w in ngram))

    if filter_stopwords:
        ngrams_ = (ngram for ngram in ngrams_
                   if not any(w.is_stop for w in ngram))
    if good_tags:
        ngrams_ = (ngram for ngram in ngrams_
                   if all(w.pos_ in good_tags for w in ngram))

    for ngram in ngrams_:
        yield ngram


def normalize(term):
    """
    Parses a token or span of tokens into a lemmatized string.
    Proper nouns are not lemmatized.

    Args:
        term: a spacy.Token or spacy.Span object
    Returns:
        lemmatized string
    Raises:
        TypeError if input is not a Token or Span
    """
    if isinstance(term, spacy.tokens.token.Token):
        return term.text if term.pos_ == 'PROPN' else term.lemma_
    elif isinstance(term, spacy.tokens.span.Span):
        return ' '.join(word.text if word.pos_ == 'PROPN' else word.lemma_
                        for word in term)
    else:
        msg = "Normalization requires a Token or Span, not {}.".format(type(term))
        raise TypeError(msg)


def top_keys(n, dict):
    # Helper function for retrieving top n keys in a dictionary
    return sorted(dict.keys(), key=lambda k: dict[k], reverse=True)[:n]



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
