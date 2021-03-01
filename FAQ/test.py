from transformers import BertTokenizer, TFBertModel
import pandas
import os
from FAQ.faq_match import FAQ_match
faq_match = FAQ_match()
if __name__ == "__main__":
    faq_match = FAQ_match()
    # print(os.path.dirname(__file__))
    # print(os.getcwd())
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