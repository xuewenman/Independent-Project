import jieba
import jieba.analyse
import six
import re
import operator
import io
import jieba.posseg as pseg


def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False


def load_stop_words(stop_word_file):
    """
    Utility function to load stop words from a file and return as a list of words
    @param stop_word_file Path and file name of a file containing stop words.
    @return list A list of stop words.
    """
    stop_words = []
    for line in open(stop_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                stop_words.append(word)
    #stop_words.reverse()
    stop_words.append(' ')
    return stop_words


def separate_words(text, min_word_return_size):
    """
    Utility function to return a list of all words that are have a length greater than a specified number of characters.
    @param text The text that must be split in to words.
    @param min_word_return_size The minimum no of characters a word must have to be included.
    """
    word_list = jieba.cut(text, cut_all=False)
    words = []
    for single_word in word_list:
        current_word = single_word.strip()
        # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
        if len(current_word) > min_word_return_size and current_word != '' and not is_number(current_word):
            words.append(current_word)
    return words


def split_sentences(text):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    #
    delimiters = ".", ",", ";", ":", "?", "!", \
                 "。", "，", "？", "！", "；", "：", \
                 "（", "）", "“", "”"
    regexPattern = '|'.join(map(re.escape, delimiters))
    sentences = re.split(regexPattern, text)
    return sentences


def build_stop_word_regex(stop_word_list):
    stop_word_regex_list = []
    for word in stop_word_list:
        word_regex = word
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile('|'.join(map(re.escape, stop_word_regex_list)))
    return stop_word_pattern


def generate_candidate_keywords(sentence_list, stop_word_pattern):
    phrase_list = []
    for s in sentence_list:
        tmp = re.sub(stop_word_pattern, '|', s.strip())
        tmp = re.sub(stop_word_pattern, '|', tmp)
        phrases = tmp.split("|")
        for phrase in phrases:
            phrase = phrase.strip()
            if phrase != "" or len(phrase)>1:
                phrase_list.append(phrase)
    return phrase_list


def check_rule(phrase):
    words_characters = pseg.cut(phrase)
    for word, flag in words_characters:
        pass
    if flag == 'v':
        return False
    else:
        return True


def generate_candidate_keywords_jieba(sentence_list):
    phrase_list = []
    for s in sentence_list:
        stop_word_list = []
        words_characters = pseg.cut(s)
        for word, flag in words_characters:
            if flag not in ['an', 'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn', 'j'] or len(word) == 1:
                stop_word_list.append(word)
        if len(stop_word_list) > 0:
            stop_word_pattern = build_stop_word_regex(stop_word_list)
            tmp = re.sub(stop_word_pattern, '|', s.strip())
        else:
            tmp = s
        #print(tmp)
        phrases = tmp.split('|')
        for phrase in phrases:
            if phrase != "" and len(phrase) > 1:
                phrase_list.append(phrase)
    return phrase_list


def calculate_word_scores(phraseList):
    word_frequency = {}
    word_degree = {}
    for phrase in phraseList:
        word_list = separate_words(phrase, 0)
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        # if word_list_degree > 3: word_list_degree = 3 #exp.
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            word_degree[word] += word_list_degree  # orig.
            # word_degree[word] += 1/(word_list_length*1.0) #exp.
    for item in word_frequency:
        word_degree[item] = word_degree[item] + word_frequency[item]

    # Calculate Word scores = deg(w)/frew(w)
    word_score = {}
    for item in word_frequency:
        word_score.setdefault(item, 0)
        word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)  # orig.
    # word_score[item] = word_frequency[item]/(word_degree[item] * 1.0) #exp.
    return word_score


def get_tfidf_dict(phraseList):
    total_len = 0
    word_freq = {}
    phrase_freq = {}
    for phrase in phraseList:
        word_list = separate_words(phrase, 0)
        total_len += float(len(word_list))
        for word in word_list:
            word_freq.setdefault(word, 0)
            word_freq[word] += 1
    for key in word_freq:
        word_freq[key] *= jieba.analyse.get_idf_jieba(key)/total_len
    ###
    for phrase_ in phraseList:
        phrase_score = 0
        phrase_freq.setdefault(phrase_, 0)
        word_list_ = separate_words(phrase_, 0)
        phrase_len = len(phrase_)
        for word_ in word_list_:
            phrase_score += word_freq[word_]
        phrase_freq[phrase_] = phrase_score / phrase_len
    #return phrase_freq
    return word_freq


def generate_candidate_keyword_scores(phrase_list, word_score, min_keyword_frequency=1):
    keyword_candidates = {}
    for phrase in phrase_list:
        keyword_candidates.setdefault(phrase, 0)
        word_list = separate_words(phrase, 0)
        candidate_score = 0
        for word in word_list:
            candidate_score += word_score[word]
        keyword_candidates[phrase] = candidate_score
    return keyword_candidates


class Rake(object):
    def __init__(self, stop_words_path):
        self.__stop_words_path = stop_words_path
        self.__stop_words_list = load_stop_words(stop_words_path)

    def run(self, text):
        sentence_list = split_sentences(text)
        stop_words_pattern = build_stop_word_regex(self.__stop_words_list)
        #phrase_list = generate_candidate_keywords(sentence_list, stop_words_pattern)
        phrase_list = generate_candidate_keywords_jieba(sentence_list)
        #word_scores = calculate_word_scores(phrase_list)
        #keyword_candidates = generate_candidate_keyword_scores(phrase_list, word_scores)
        keyword_candidates = get_tfidf_dict(phrase_list)
        sorted_keywords = sorted(six.iteritems(keyword_candidates), key=operator.itemgetter(1), reverse=True)
        #return sorted_keywords
        return keyword_candidates

    def run2(self, text):
        sentence_list = split_sentences(text)
        phrase_list = generate_candidate_keywords_jieba(sentence_list)
        tfidf_dict = get_tfidf_dict(phrase_list)


def train_candidate_features(candidates, title, content):
    tags_features = []
    for candidate in candidates:
        doc_len = float(len(content))
        first_match = content.find(candidate)
        last_match = content.rfind(candidate)
        spread = (last_match - first_match)/doc_len
        in_title = 1 if candidate in title else 0
        #in_summary = 1 if candidate in summary else 0


"""
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

"""

if __name__ ==  '__main__':
    #stop_path = "SmartStoplist.txt"
    #rake_object = rake.Rake(stop_path, 5, 3, 4)
    #words = rake.separate_words('I am not a hero +', 0)
    #for word in words:
    #    print(word)
    #test_chinese_news = '1.txt'
    #file_in = open(test_chinese_news, 'r')
    sample_file = io.open("11.txt", 'r')
    test_content = sample_file.read()
    #sentences = split_sentences(test_content)
    #for sentence in sentences:
    #    print(sentence)

    stop_words_path = 'cn_stopword_list_1208.txt'
    #stop_words = load_stop_words(stop_words_path)
    #words = separate_words("让外交名片实至名归", 0)
    #for word in words:
    #    print(word)

    rake_object = Rake(stop_words_path)
    sorted_keywords = rake_object.run(test_content)
    for key, value in sorted_keywords:
        print(key, value)
    """
    #print(separate_words("让外交名片实至名归", 0))
    tags = jieba.analyse.extract_tags(test_content, topK=10, withWeight=True)
    print(tags)

    freq = jieba.analyse.get_tfidf_all_jieba(test_content)
    for item in freq:
        print(item, freq[item])

    # read json data from file
    dicts_from_files = []
    with open('KeywordPost_mini.json', 'r') as info:
        for line in info:
            dicts_from_files.append(eval(line))


    dicts_features = []
    for i in range(len(dicts_from_files)):
        tags = dicts_from_files[i]['tags']
        title = dicts_from_files[i]['title']
        content = dicts_from_files[i]['content']
        #summary = dicts_from_files[i]['summary']


    #print(jieba.analyse.get_max_min_idf())
    """