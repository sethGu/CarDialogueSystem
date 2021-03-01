import os
import codecs
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from FAQ.keyword_processer import KeywordProcesser
from FAQ.BertVector import BertVector

# faq_model_dir = '../data/faq'
# faq_file = os.path.join(faq_model_dir, 'faq_corpus.xlsx')
faq_file = r'C:\Users\xuchanghua\PycharmProjects\rasa_faq\data\faq\faq_corpus.xlsx'

# stop_word_file = os.path.join(faq_model_dir, 'stop_word.csv')
stop_word_file = r'C:\Users\xuchanghua\PycharmProjects\rasa_faq\data\faq\stop_word.csv'
threshold = 0.9


def rm_stword(sentence, stword):
    '''
    这个是用来去除停用词的
    '''
    #     pattern = u'[a-zA-Z0-9]+'
    dt_list = []
    new_line = sentence.strip().split()  #切分词
    for word in new_line:
        #         len_th = re.findall(pattern,word) #len_th 为列表
        #re.findall在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果没有找到匹配的，则返回空列表
        if word not in stword:
            dt_list.append(word)
    return ' '.join(dt_list)


def cosine_distance(matrix1, matrix2):
    '''
    这个函数应该是用来计算语义相似度的，使用的是余弦相似度相似的函数来求
    '''
    # print(matrix1.shape)
    # print(matrix2.shape)
    matrix1_matrix2 = np.dot(matrix1, matrix2.transpose())
    matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
    matrix1_norm = matrix1_norm[:, np.newaxis]
    matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum())
    matrix2_norm = [matrix2_norm]
    matrix2_norm = np.array(matrix2_norm)
    #matrix2_norm = matrix2_norm[:, np.newaxis]
    cosine_distance = np.divide(matrix1_matrix2, np.dot(matrix1_norm, matrix2_norm.transpose()))
    return cosine_distance


class FAQ_match(object):
    def __init__(self):
        '''faq_file: faq 问题答案'''
        self.faq_question = pd.read_excel(faq_file)
        with codecs.open(stop_word_file, 'r', 'utf-8') as f:
            self.st_w = [line.strip() for line in f]
        self.cv = CountVectorizer()  #创建词袋数据结构
        self.alpha = threshold  #设置相似度的阈值
        question_list = []
        self.faq_dict = {}  #加载问题和答案
        self.faq_qus = []  #加载问题
        self.faq_ans = []  #加载答案
        self.sk2_flag = ['skii', 'sk2', 'sk-ii', 'SK-II', 'sk-II', 'SK-2', 'sk-2']


        self.faq_vec = np.array(question_list)  #加载去掉主语之后的faq矩阵
        # print(len(self.faq_vec))
        # print(self.faq_vec)
        # print(self.faq_vec.shape)
        entity_dict = {}  #加载主语抽取的词库
        with codecs.open(r'C:\Users\xuchanghua\PycharmProjects\rasa_faq\data\faq\skii_entity.csv', 'r', 'utf-8') as reader:
            for line in reader.readlines():
                entity_dict[line.strip()] = line.strip()
        self.kp = KeywordProcesser()
        self.kp.add_keyword_from_dict(entity_dict)  #加载抽取的词库
        # print(self.kp.keyword_trie_dict)
        pd_dt = pd.read_csv(r'C:\Users\xuchanghua\PycharmProjects\rasa_faq\data\faq\skii_synonym.csv')  #加载同义词库
        pd_group = pd_dt.groupby(['seed_word'])['similar_word'].apply(lambda x: x.tolist())
        # print(pd_group)
        self.word2id = {}
        self.id2word = {}
        for i in range(0, len(pd_group)):
            wd_list = []
            self.word2id[pd_group.index[i]] = i
            wd_list.append(pd_group.index[i])
            for wd in pd_group[i]:
                self.word2id[wd] = i
                wd_list.append(wd)
            self.id2word[i] = wd_list
        # print(self.word2id)
        # print(self.id2word)
    def search_sym(self, kw_1, kw_2):
        '''查询两个词是否为同义词,是同义词返回1，否则返回0'''
        flag = 0
        kw_1_flag = -1
        kw_2_flag = -1
        kw_1 = kw_1.strip().replace(' ', '').lower()
        kw_2 = kw_2.strip().replace(' ', '').lower()
        if kw_1 == kw_2:
            flag = 1
        try:
            kw_1_flag = self.word2id[kw_1]
        except:
            pass

        try:
            kw_2_flag = self.word2id[kw_2]
        except:
            pass
        if kw_1_flag != -1 and kw_2_flag != -1 and kw_1_flag == kw_2_flag:
            flag = 1
        return flag


    def judge_similar(self, question, qs_list, ans_list):
        '''找到最终匹配出来的问题'''
        qus_all_cut = []
        qus_cut = ' '.join(jieba.cut(question, cut_all=True))
        qus_cut = rm_stword(qus_cut, self.st_w)
        qus_all_cut.append(qus_cut)
        for eachline in qs_list:
            line_cut = rm_stword(' '.join(jieba.cut(eachline, cut_all=True)), self.st_w)
            qus_all_cut.append(line_cut)
        cv_fit = self.cv.fit_transform(qus_all_cut)
        cv_fit = cv_fit.toarray()
        qus_vec = cv_fit[0, :]
        qus_vec = qus_vec.reshape(1, qus_vec.shape[0])
        qus_search_vec = cv_fit[1:, :]
        cos_dc = cosine_distance(qus_vec, qus_search_vec).tolist()[0]
        max_index = cos_dc.index(max(cos_dc))
        sim_qus = qs_list[max_index]
        sim_ans = ans_list[max_index]

        return (sim_qus, sim_ans)




if __name__ == "__main__":
    faq_match = FAQ_match()
    # print(faq_match.faq_qus)
    # print(len(faq_match.faq_qus))
    # print(faq_match.faq_ans)
    # print(len(faq_match.faq_ans))

    while (True):
        message = input('Enter your question:')
        if message != 'quit':
            #             time1 = datetime.datetime.now()
            result = faq_match.faq_match(message)
            #result = faq_match.judge_similar(message, faq_match.faq_qus, faq_match.faq_ans)
            print(result)
#             time2= datetime.datetime.now()
#             aa = time2 - time1
#             print (aa)
        else:
            break
