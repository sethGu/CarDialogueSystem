import os
import codecs
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from FAQ.keyword_processer import KeywordProcesser
from FAQ.BertVector import BertVector

# faq_model_dir = '../data/faq_data'
# faq_file = os.path.join(faq_model_dir, 'faqwithoutEntiy.xlsx')

path = os.path.dirname(__file__).split('/')[:-1]
newpath = '/'.join(path)
faq_model_dir = os.path.join(newpath,'data/faq_data')
faq_file = os.path.join(newpath,'data/faq_data/faqwithoutEntiy.xlsx')
# faq_file = r'E:\dialogue system\CarDialogueSystem\data\faq_data\faqwithoutEntiy.xlsx'
# faq_file = r'C:\Users\xuchanghua\PycharmProjects\rasa_faq\data\faq\faq_corpus.xlsx'
#################################################################################


stop_word_file = os.path.join(faq_model_dir, 'stop_word.csv')

# stop_word_file = r'E:\dialogue system\CarDialogueSystem\data\faq_data\stop_word.csv'
#################################################################################

bv = BertVector()
threshold = 0.8


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

        for index, row in self.faq_question.iterrows():

            self.faq_dict[row['question']] = row['answer']
            self.faq_qus.append(row['question'])
            self.faq_ans.append(row['answer'])

            # question_list.append(bv.encode(row['question_bak'], 16).tolist())
        ###################################################
        # 将转化的词向量写入txt
        # with open(r'E:\anew\文档\rasateat\上汽数据\sentence_vect.txt','a') as f:
        #     for vect in question_list:
        #         print(vect)
        #         for flot in vect:
        #             f.write(str(flot))
        #             f.write(',')
        #         f.write('\n')
        #####################################################
        # 读取文件中的句向量

        sentence_file = os.path.join(faq_model_dir,'sentence_vect.txt')
        # sentence_file = r'E:\dialogue system\CarDialogueSystem\data\faq_data\sentence_vect.txt'
        with open(sentence_file) as f:
            lines = f.readlines()
            for line in lines:
                line.strip()
                line_new = line.split(',')
                line_new = line_new[:-1]
                cov_line = []
                for i in line_new:
                    cov_line.append(float(i))
                question_list.append(cov_line)
            # print(bv.encode(row['question_bak'], 16).tolist())
        self.faq_vec = np.array(question_list)  #加载去掉主语之后的faq矩阵
        # print(len(self.faq_vec))
        # print(self.faq_vec)
        # print(self.faq_vec.shape)
        entity_dict = {}  #加载主语抽取的词库
        # with codecs.open(r'C:\Users\xuchanghua\PycharmProjects\rasa_faq\data\faq\skii_entity.csv', 'r', 'utf-8') as reader:
        ##########################################################################################################

        entity_file = os.path.join(faq_model_dir,'shangqi_entity.csv')
        # entity_file = r'E:\dialogue system\CarDialogueSystem\data\faq_data\shangqi_entity.csv'
        with codecs.open(entity_file, 'r',
                         'utf-8') as reader:
            for line in reader.readlines():
                entity_dict[line.strip()] = line.strip()
        self.kp = KeywordProcesser()
        self.kp.add_keyword_from_dict(entity_dict)  #加载抽取的词库
        # print(self.kp.keyword_trie_dict)
        # pd_dt = pd.read_csv(r'C:\Users\xuchanghua\PycharmProjects\rasa_faq\data\faq\skii_synonym.csv')  #加载同义词库
        #######################################################################################################

        synonym_file = os.path.join(faq_model_dir,'shangqi_synonym.csv')

        # synonym_file = r'E:\dialogue system\CarDialogueSystem\data\faq_data\shangqi_synonym.csv'

        pd_dt = pd.read_csv(synonym_file)

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

    def faq_qus_choose(self, question, qus_list):
        '''返回与问题主语一致的问题,返回与主语一致的句子索引'''
        #         flag_str = ['小灯泡','小红瓶','小银瓶','大红瓶','男友面膜','神仙水']
        object_list = []
        sk2_flag = 0
        qus_kw_left = []
        qus_kw = self.kp.extract_keywords(question)
        # print('qus_kw',end='')
        # print(qus_kw)
        if len(qus_kw) == 0:
            qus_kw_left = qus_kw
        else:
            for qp_list in qus_kw:
                if qp_list[2] not in self.sk2_flag:
                    qus_kw_left.append(qp_list[2])

        qus_sim_kw = []
        for eachline in qus_list:
            kw = self.kp.extract_keywords(eachline)
            kw_left = []
            if len(kw) == 0:
                kw_left = kw
            else:
                for kw_list in kw:
                    if kw_list[2] not in self.sk2_flag:
                        kw_left.append(kw_list)
            qus_sim_kw.append(kw_left)

        entity_same_list = []
        ###############
        # print('qus_kw_left',end='')
        # print(qus_kw_left)
        if len(qus_kw_left) == 0:
            for i in range(0, len(qus_list)):
                object_list.append(i)
                if len(qus_sim_kw[i]) == 0:
                    entity_same_list.append(i)

        if len(qus_kw_left) == 1:
            for i in range(0, len(qus_list)):
                if len(qus_sim_kw[i]) == 0:
                    object_list.append(i)
                if len(qus_sim_kw[i]) == 1 and self.search_sym(qus_kw[0][2], qus_sim_kw[i][0][2]) == 1:
                    object_list.append(i)
                    entity_same_list.append(i)

        if len(qus_kw_left) >= 2:
            for i in range(0, len(qus_list)):
                if len(qus_sim_kw[i]) >= 1:
                    for q_list in qus_sim_kw[i]:
                        for qus_sim_list in qus_kw_left:  #
                            if self.search_sym(q_list[2], qus_sim_list) == 1:
                                object_list.append(i)
                                entity_same_list.append(i)
        object_list = list(set(object_list))
        entity_same_list = list(set(entity_same_list))
        return (object_list, entity_same_list)

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

    def faq_match(self, question):
        '''
        :param question: 需要匹配的问题
        :return: faq_result: 存在：[(问题，答案)]；不存在：[]
        '''
        results = []
        qs_list = []
        ans_list = []
        faq_result = []
        kw_list = self.kp.extract_keywords(question)
        # print(kw_list)
        qus_rm_entity = question
        if len(kw_list) == 0:
            qus_rm_entity = question
        else:
            for kw in kw_list:
                qus_rm_entity = qus_rm_entity.replace(kw[2], '')
            qus_rm_entity = qus_rm_entity.lstrip('的').replace('啊', '').replace('呀', '').replace('咩', '')
        # print(qus_rm_entity)
        qus_vec = bv.encode(qus_rm_entity, 16)
        sim_list = cosine_distance(self.faq_vec, qus_vec)
        # print(sim_list)
        prob_list = []
        for i in range(0, len(sim_list)):
            if sim_list[i] >= self.alpha:
                qs_list.append(self.faq_qus[i])
                ans_list.append(self.faq_ans[i])
                prob_list.append(sim_list[i])
        if len(qs_list) == 0:
            faq_result = []
        if len(qs_list) != 0:
            sim_qs_list = []
            sim_ans_list = []
            sim_prob_list = []
            (qus_index, entity_same) = self.faq_qus_choose(question, qs_list)
            if len(qus_index) == 0:
                faq_result = []

            if len(qus_index) == 1:
                faq_result = [qs_list[qus_index[0]], ans_list[qus_index[0]]]
            if len(qus_index) > 1:
                # 对prob进行排序
                sorted_prob = []
                for index in qus_index:
                    sorted_prob.append(prob_list[index])
                sorted_prob = sorted(sorted_prob, reverse=True)
                if sorted_prob[0] == sorted_prob[1]:
                    for no in entity_same:
                        sim_qs_list.append(qs_list[no])
                        sim_ans_list.append(ans_list[no])
                        sim_prob_list.append(prob_list[no])
                    max_no = sim_prob_list.index(max(sim_prob_list))
                    faq_result = [sim_qs_list[max_no], sim_ans_list[max_no]]
                else:
                    for no in qus_index:
                        sim_qs_list.append(qs_list[no])
                        sim_ans_list.append(ans_list[no])
                        sim_prob_list.append(prob_list[no])
                    max_no = sim_prob_list.index(max(sim_prob_list))
                    faq_result = [sim_qs_list[max_no], sim_ans_list[max_no]]
            print(faq_result)
        return faq_result

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
            # print(result[1])
#             time2= datetime.datetime.now()
#             aa = time2 - time1
#             print (aa)
        else:
            break
