import itertools
import nltk
import string
import io
import gensim
import networkx
import _collections
import math
import re


def print_result(list_):
    for li in list_:
        print(li)


def print_result2(list_):
    for li in list_:
        for tup in li:
            print(tup)


def extract_candidate_chunks(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):

    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize, Pos-tag words and chunk using regular expression
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group).lower()
                 for key, group in itertools.groupby(all_chunks, lambda w_p_chunk: w_p_chunk[2] != 'O') if key]
    """
    groups = itertools.groupby(all_chunks, lambda w_p_chunk: w_p_chunk[2] != 'O')
    for key, group in groups:
        if key:
            for g in group:
                print(g)
    candidates = [(key, group) in itertools.groupby(all_chunks, lambda w_p_chunk: w_p_chunk[2] != 'O') if key]
    test_array = range(123)
    test = [''.join(str(num) for num in group)
             for key, group in itertools.groupby(test_array, key=lambda x: x%3 != 0) if key]

    """
    return [cand for cand in candidates if cand not in stop_words and not all(char in punct for char in cand)]


def extract_candidate_words(text, good_tags=set(['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS'])):
    import itertools, nltk, string

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize and POS-tag words
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))
    # filter on certain POS tags and lowercase all words
    candidates = [word.lower() for word, tag in tagged_words
                  if tag in good_tags and word.lower() not in stop_words
                  and not all(char in punct for char in word)]

    return candidates


def score_keyphrase_by_tfidf(texts, candidates='chunks'):
    if candidates == 'chunks':
        boc_texts = [extract_candidate_chunks(text) for text in texts]
    elif candidates == 'words':
        boc_texts = [extract_candidate_words(text) for text in texts]
    dictionary = gensim.corpora.Dictionary(boc_texts)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_texts]
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    return corpus_tfidf, dictionary


def score_keyphrase_by_textrank(text, n_keywords=0.05):
    words = [word.lower()
             for sent in nltk.sent_tokenize(text)
             for word in nltk.word_tokenize(sent)]
    candidates = extract_candidate_words(text)
    #build graph
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))
    def pairwise(iterable):
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)
    for w1, w2 in pairwise(candidates):
        if w2:
            graph.add_edge(*sorted([w1, w2]))
    ranks = networkx.pagerank(graph)
    if 0 < n_keywords < 1:
        n_keywords = int(round(len(candidates) * n_keywords))
    word_ranks = {word_rank[0]: word_rank[1]
                  for word_rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:n_keywords]}
    keywords = set(word_ranks.keys())
    keyphrases = {}
    j = 0
    for i, word in enumerate(words):
        if i < j:
            continue
        if word in keywords:
            kp_words = list(itertools.takewhile(lambda x: x in keywords, words[i:i + 10]))
            avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
            keyphrases[' '.join(kp_words)] = avg_pagerank
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)

    return sorted(keyphrases.items(), key=lambda x: x[1], reverse=True)


def extract_candidate_feature(candidates, doc_text, doc_excerpt, doc_title):
    candidate_scores = _collections.OrderedDict()
    # 计数器
    doc_word_counts = _collections.Counter(word.lower()
                                          for sent in nltk.sent_tokenize(doc_text)
                                          for word in nltk.word_tokenize(sent))

    for candidate in candidates:
        pattern = re.compile(r'\b'+re.escape(candidate)+r'(\b|[,;.!?]|\s)',re.IGNORECASE)
        cand_doc_count = len(pattern.findall(doc_text))
        if not cand_doc_count:
            print
            '**WARNING:', candidate, 'not found!'
            continue
        # 统计学
        candidate_words = candidate.split()
        max_word_length = max(len(w) for w in candidate_words)
        term_length = len(candidate_words)
        sum_doc_word_counts = float(sum(doc_word_counts[w] for w in candidate_words))
        try:
            # lexical cohesion doesn't make sense for 1-word terms
            if term_length == 1:
                lexical_cohesion = 0.0
            else:
                lexical_cohesion = term_length * (
                1 + math.log(cand_doc_count, 10)) * cand_doc_count / sum_doc_word_counts
        except (ValueError, ZeroDivisionError) as e:
            lexical_cohesion = 0.0

        # positional
        # found in title, key excerpt
        in_title = 1 if pattern.search(doc_title) else 0
        in_excerpt = 1 if pattern.search(doc_excerpt) else 0
        # first/last position, difference between them (spread)
        doc_text_length = float(len(doc_text))
        first_match = pattern.search(doc_text)
        abs_first_occurrence = first_match.start() / doc_text_length
        if cand_doc_count == 1:
            spread = 0.0
            abs_last_occurrence = abs_first_occurrence
        else:
            for last_match in pattern.finditer(doc_text):
                pass
            abs_last_occurrence = last_match.start() / doc_text_length
            spread = abs_last_occurrence - abs_first_occurrence
        candidate_scores[candidate] = {'term_count': cand_doc_count,
                                       'term_length': term_length, 'max_word_length': max_word_length,
                                       'spread': spread, 'lexical_cohesion': lexical_cohesion,
                                       'in_excerpt': in_excerpt, 'in_title': in_title,
                                       'abs_first_occurrence': abs_first_occurrence,
                                       'abs_last_occurrence': abs_last_occurrence}

    return candidate_scores


if __name__ == '__main__':
    text = 'Health is the level of functional or metabolic efficiency of a living organism. ' \
           'In humans it is the ability of individuals or communities to adapt and self-manage when facing physical, ' \
           'mental or social challenges. The most widely accepted definition of good health is that of ' \
           'the World Health Organization Constitution. It states:"health is a state of complete physical, ' \
           'mental and social well-being and is not merely the absence of disease or infirmity" ' \
           '(World Health Organization, 1946). In more recent years, this statement has been amplified to include ' \
           'the ability to lead a "socially and economically productive life." The WHO definition is not without ' \
           'criticism, mainly that it is much too broad.'
    texts = []
    text_path = '10.txt'
    text_flag = io.open(text_path, 'r')
    text_content = text_flag.read()
    texts.append(text_content)
    text_path = '9.txt'
    text_flag = io.open(text_path, 'r')
    text_content = text_flag.read()
    texts.append(text_content)

    corpus_tfidf, dictionary = score_keyphrase_by_tfidf(texts, 'chunks')
    #output = extract_candidate_words(text)
    #print(output.__sizeof__())
    #for text_ in text:
    #    print(text_)
    #print_result(corpus_tfidf)
    output = score_keyphrase_by_textrank(text)
    print(output)



