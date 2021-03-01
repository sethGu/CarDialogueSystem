from FAQ.model import car_aq_model
from transformers import BertTokenizer
import tensorflow as tf
import numpy
import os
path = os.path.dirname(__file__).split('/')[:-1]
newpath = '/'.join(path)
vocab_file = os.path.join(newpath,'bert_ch/vocab.txt')

# vocab_file = bert_modle_vocab + r'\vocab.txt'
# vocab_file = r'E:\dialogue system\CarDialogueSystem\bert_ch\vocab.txt'
# vocab_file = '../bert_ch/vocab.txt'
class BertVector:
    def __init__(self):
        self.tokenizer = BertTokenizer(vocab_file)
        self.model = car_aq_model()

    def encode(self, sentence, max_sentence_len):
        bert_input = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_sentence_len,
            padding='max_length',
            # return_attention_mask=True
        )
        input_ids = tf.convert_to_tensor([bert_input['input_ids']])
        token_type_ids = tf.convert_to_tensor([bert_input['token_type_ids']])
        attention_mask = tf.convert_to_tensor([bert_input['attention_mask']])
        outputs = self.model(input_ids,token_type_ids,attention_mask)

        return outputs.numpy()
if __name__ == '__main__':
    bv = BertVector()
    message = '今天天气怎么样'
    message_vec = bv.encode(message, 16)
    print(message_vec)

