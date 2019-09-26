from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
'''

SOS_token = 0   # start-of-string token
EOS_token = 1   # end-of-string token

#---------------------Data Preparation--------------------------#

class Lang:
    '''
    语言类，计算word的index和count
    '''
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# 将Unicode的word转换为ASCII
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s



def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    # List Comprehensions
    # 语法：
    #
    # 　　[expression for iter_val in iterable]
    #
    # 　　[expression for iter_val in iterable if cond_expr]

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs



MAX_LENGTH = 10     # the max length of sentences we use

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

#---------------------Seq2Seq Model--------------------------#
# With a seq2seq model the encoder creates a single vector which,
# in the ideal case, encodes the “meaning” of the input sequence into a single vector — a single point in some N dimensional space of sentences.

class EncoderRNN(nn.Module):
    '''
    Encoder 将每一个输入的word都encode为一个向量+一个hidden state,hidden state将作为下一个word的输入
    '''
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # 初始化一个word embedding器
        self.embedding = nn.Embedding(input_size, hidden_size)   # agr1: num_embeddings, 代表字典长度, agr2: embedding_dim，每个embedding向量的长度

        # 初始化一个 gated recurrent unit
        self.gru = nn.GRU(hidden_size, hidden_size)     # arg1: input x的纬度 arg2: hidden state的纬度

    def forward(self, *input, hidden):
        embedded = self.embedding(input).view(1, 1, -1) #重新整理tensor，-1为不指名size，由前两个计算得出
        output = embedded
        output, hidden = self.gru(output, hidden)   #使用当前word的embedded和上一个word的hidden，输出当前word的output和hidden
        return output, hidden

    def initHidden(self):
        # 生成一个1*1*hidden_size的初始hidden state
        return torch.zeros(1, 1, self.hidden_size, device=device)



class DecoderRNN(nn.Module):
    '''
    Decoder: 将encoder得到的输出和hidden state作为decoder的输入.
    最简单的版本的Seq2Seq将最后一个词的hidden state称为context，代表整个句子的意思
    context vector作为decoder的initial hidden state

    initial input 为SOS: start-of-string token
    '''
    def __init__(self,  hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size


        self.embedding = nn.Embedding(output_size, hidden_size)   # agr1: num_embeddings, 代表字典长度, agr2: embedding_dim，每个embedding向量的长度
        self.gru = nn.GRU(hidden_size, hidden_size)     # arg1: input x的纬度 arg2: hidden state的纬度
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim =1)

    def forward(self, *input, hidden):
        output = self.embedding(input).view(1, 1, -1) #获得input的embedded
        output = F.relu(output)     # embedded input再经过relu
        output, hidden = self.gru(output, hidden)   # relu后的input与previous hidden一起进入GRU
        output = self.softmax(self.out(output[0]))      #对输出进行softmax
        return output, hidden

    def initHidden(self):
        # 生成一个1*1*hidden_size的初始hidden state
        return torch.zeros(1, 1, self.hidden_size, device=device)












