import jieba
import jieba.analyse
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    test_news = '1.txt'
    file_in = open(test_news, 'r')
    test_content = file_in.read()
    test_list = []
    test_list.append(test_content)
    seg_list = jieba.cut(test_content, cut_all=False)
    #for word in seg_list:
    #    print('%s' % word)
    #print(test_content)
    seg_gender = pseg.cut(test_content)
    #for word, flag in seg_gender:
    #    print('%s %s' %(word, flag))

    ## print the type of the ...
    print('输入类型%s' % type(test_content))
    print('输入类型%s' % type(seg_list))
    print('输入类型%s' % type(seg_gender))

    ##jieba TF-IDF
    ## return 'list'
    #words1 = jieba.analyse.extract_tags(test_content, withWeight=True, allowPOS=())
    #for word, freq in words1:
    #    print('%s %s' % (word, freq))
    #print('前结果：', len(words1))

    jieba.analyse.set_stop_words('chinese_stop_words.txt')
    words2 = jieba.analyse.extract_tags(test_content, withWeight=True, allowPOS=())
    for word, freq in words2:
        print('%s %s' % (word, freq))
    print('后结果：', len(words2))


    ## sklearn
    #vectorizer = CountVectorizer()
    #transformer = TfidfTransformer()
    #X = vectorizer.fit_transform(test_list)
    #print(X)
  #  for word in words:
  #      print(word)