import numpy as np
import operator
from collections import defaultdict
import logging

class Vocab(object):
    def __init__(self, unk='<unk>'):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = unk
        self.add_word(self.unknown, count=0)

        
    def add_word(self, word, count=1):
        word = word.strip()
        if len(word) == 0:
            return
        elif word.isspace():
            return
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

        
    def construct(self, words):
        for word in words:
            self.add_word(word)
        self.total_words = float(sum(self.word_freq.values()))
        print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))
 

    def limit_vocab_length(self, length):
        """
        Truncate vocabulary to keep most frequent words
        
        Args:
            None
            
        Returns:
            None 
        """
        if length > self.__len__():
            return
        new_word_to_index = {self.unknown:0}
        new_index_to_word = {0:self.unknown}
        self.word_freq.pop(self.unknown)          #pop unk word
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        vocab_tup = sorted_tup[:length]
        self.word_freq = dict(vocab_tup)
        for word in self.word_freq:
            index = len(new_word_to_index)
            new_word_to_index[word] = index
            new_index_to_word[index] = word
        self.word_to_index = new_word_to_index
        self.index_to_word = new_index_to_word
        self.word_freq[self.unknown]=0
        
        
    def save_vocab(self, filePath):
        """
        Save vocabulary a offline file
        
        Args:
            filePath: where you want to save your vocabulary, every line in the 
            file represents a word with a tab seperating word and it's frequency
            
        Returns:
            None 
        """
        self.word_freq.pop(self.unknown)
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        with open(filePath, 'wb') as fd:
            for (word, freq) in sorted_tup:
                fd.write(('%s\t%d\n'%(word, freq)).encode('utf-8'))
            

    def load_vocab_from_file(self, filePath, sep='\t'):
        """
        Truncate vocabulary to keep most frequent words
        
        Args:
            filePath: vocabulary file path, every line in the file represents 
                a word with a tab seperating word and it's frequency
            
        Returns:
            None 
        """
        with open(filePath, 'rb') as fd:
            for line in fd:
                line_uni = line.strip().decode('utf-8')
                word, freq = line_uni.split(sep)
                index = len(self.word_to_index)
                if word not in self.word_to_index:
                    self.word_to_index[word] = index
                    self.index_to_word[index] = word
                self.word_freq[word] = int(freq)
            print 'load from <'+filePath+'>, there are {} words in dictionary'.format(len(self.word_freq))
 

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    
    def decode(self, index):
        return self.index_to_word[index]

    
    def __len__(self):
        return len(self.word_to_index)

def loadId2Tag(fileName):
    """
    """
    id2tag = {}
    tag2id = {}
    with open(fileName, 'rb') as fd:
        for line in fd:
            line_uni = line.decode('utf-8')
            idTag = line_uni.split('\t')
            index = int(idTag[0])
            tag = idTag[1].strip()
            id2tag[index] = tag
            tag2id[tag]=index
    return id2tag, tag2id

def encodeNpad(dataList, vocab, trunLen=0):
    sentLen = []
    data_matrix = []
    for wordList in dataList:
        length = len(wordList)
        if trunLen !=0:
            length=min(length, trunLen)
        sentEnc = []
        if trunLen == 0:
            for word in wordList:
                sentEnc.append(vocab.encode(word))
        else:
            for i in range(trunLen):
                if i < length:
                    sentEnc.append(vocab.encode(wordList[i]))
                else:
                    sentEnc.append(vocab.encode(vocab.unknown))
        sentLen.append(length)
        data_matrix.append(sentEnc)
    return sentLen, data_matrix


def mkDataSet(fileName, num_class, vocab, tag_vocab, num_steps):
    """
    Make data set from list of which have element structed as ([1, 3, 4], [word1, word2])

    Args:
        label2text: a list of tuple with structure of ([1, 3, 4], [word1, word2])
        num_class: number of classes
        vocab: vocabulary
        num_steps: pad to num_steps 

    Returns:
        label_matrix: a list of label tuple structed as ((0, 1, 0, 1, 0, 0, ..), ...)
        data_matrix : a list of data item. **note - item in data_matrix 
                        should correspond to item label_matrix
    """
    def loadData(fileName):
        label2text = []
        with open(fileName, 'rb') as fd:
            for line in fd:
                line_uni = line.decode('utf-8')
                idTag = line_uni.split('\t')
                labels = idTag[0].split('|')
                text = ' '.join(idTag[1:]).split()
                label2text.append((labels, text))
        return label2text
    
    label2text = loadData(fileName)
    label_matrix = []
    data_matrix = []
    for (labels, text) in label2text:
        label_vec = [0]* num_class
        for label in labels:
            label_vec[tag_vocab.encode(label)]=1
        label_matrix.append(label_vec)
        data_matrix.append(text)
    sentLen, data_matrix = encodeNpad(data_matrix, vocab, num_steps)
    return label_matrix, sentLen, data_matrix

"""Prediction """
def pred_from_prob_single(prob_matrix):
    """

    Args:
        prob_matrix: probability matrix have the shape of (data_num, class_num), 
            type of float. Generated from softmax activation
            
    Returns:
        ret: return class ids, shape of(data_num,)
    """
    ret = np.argmax(prob_matrix, axis=1)
    return ret

def pred_from_prob_multi(prob_matrix, label_num):
    """

    Args:
        prob_matrix: probability matrix have the shape of (data_num, class_num), 
            type of float. Generated from softmax activation
        label_num: specify how much positive class to pick, have the shape of (data_num), type of int

    Returns:
        ret: for each case, set all positive class to 1, shape of(data_num, class_num)
    """
    order = np.argsort(prob_matrix,axis=1)
    ret = np.zeros_like(prob_matrix, np.int32)
    
    for i in range(len(label_num)):
        ret[i][order[i][-label_num[i]:]]=1
    return ret

def pred_from_prob_sigmoid(prob_matrix, threshold=0.5):
    """
    Load tag from file

    Args:
        prob_matrix: probability matrix have the shape of (data_num, class_num), 
            type of float. Generated from sigmoid activation
        threshold: when larger than threshold, consider it as true or else false
    Returns:
        ret: for each case, set all positive class to 1, shape of(data_num, class_num)
    """
    np_matrix = np.array(prob_matrix)
    ret = (np_matrix > threshold)*1
    return ret

def calculate_accuracy_single(pred_ids, label_ids):
    """
    Args:
        pred_ids: prediction id list shape of (data_num, ), type of int
        label_ids: true label id list, same shape and type as pred_ids

    Returns:
        accuracy: accuracy of the prediction, type float
    """
    if np.ndim(pred_ids) != 1 or np.ndim(label_ids) != 1:
        raise TypeError('require rank 1, 1. get {}, {}'.format(np.rank(pred_ids), np.rank(label_ids)))
    if len(pred_ids) != len(label_ids):
        raise TypeError('first argument and second argument have different length')

    accuracy = np.mean(np.equal(pred_ids, label_ids))
    return accuracy

def calculate_accuracy_multi(pred_matrix, label_matrix):
    """
    Args:
        pred_matrix: prediction matrix shape of (data_num, class_num), type of int
        label_matrix: true label matrix, same shape and type as pred_matrix

    Returns:
        accuracy: accuracy of the prediction, type float
    """
    if np.ndim(pred_matrix) != 2 or np.ndim(label_matrix) != 2:
        raise TypeError('require rank 2, 2. get {}, {}'.format(np.rank(pred_matrix), np.rank(label_matrix)))
    if len(pred_matrix) != len(label_matrix):
        raise TypeError('first argument and second argument have different length')

    match = [np.array_equal(pred_matrix[i], label_matrix[i]) for i in range(len(label_matrix))]

    return np.mean(match)

def flatten(li):
    ret = []
    for item in li:
        if isinstance(item, list) or isinstance(item, tuple):
            ret += flatten(item)
        else:
            ret.append(item)
    return ret

"""Read and make embedding matrix"""
def readEmbedding(fileName):
    """
    Read Embedding Function
    
    Args:
        fileName : file which stores the embedding
    Returns:
        embeddings_index : a dictionary contains the mapping from word to vector
    """
    embeddings_index = {}
    with open(fileName, 'r') as f:
        for line in f:
            line_uni = line.decode('utf-8')
            values = line_uni.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def mkEmbedMatrix(embed_dic, vocab_dic):
    """
    Construct embedding matrix
    
    Args:
        embed_dic : word-embedding dictionary
        vocab_dic : word-index dictionary
    Returns:
        embedding_matrix: return embedding matrix
    """
    if type(embed_dic) is not dict or type(vocab_dic) is not dict:
        raise TypeError('Inputs are not dictionary')
    if len(embed_dic) < 1 or len(vocab_dic) <1:
        raise ValueError('Input dimension less than 1')
    
    EMBEDDING_DIM = len(embed_dic.items()[0][1])
    embedding_matrix = np.zeros((len(vocab_dic) + 1, EMBEDDING_DIM), dtype=np.float32)
    for word, i in vocab_dic.items():
        embedding_vector = embed_dic.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
 
"""Make mask""" 
def makeMask(steps, lengths):
    """
    Make a embedding mask, meant to mask out those paddings
    
    Args:
        steps: step size
        lengths: lengths
    Returns:
        ret: mask matrix, type ndarray
    
    """
    ret = np.zeros([len(lengths), steps])
    for i in range(len(lengths)):
        ret[i, :lengths[i]]=1
    return ret

"""Data iterating"""
def data_iter(data_x, data_y, len_list, batch_size):
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    len_list = np.array(len_list)
    
    data_len = len(data_x)
    epoch_size = data_len // batch_size
    
    idx = np.arange(data_len)
    np.random.shuffle(idx)
    
    for i in xrange(epoch_size):
        indices = range(i*batch_size, (i+1)*batch_size)
        indices = idx[indices]
        ret_x, ret_y, ret_len = data_x[indices], data_y[indices], len_list[indices]
        yield (ret_x, ret_y, ret_len)

def pred_data_iter(data_x, len_list, batch_size):
    data_x = np.array(data_x)
    len_list = np.array(len_list)
    
    data_len = len(data_x)
    epoch_size = data_len // batch_size

    
    for i in xrange(epoch_size):
        ret_x, ret_len = data_x[i*batch_size: (i+1)*batch_size], len_list[i*batch_size: (i+1)*batch_size]
        yield (ret_x, ret_len)
    if epoch_size*batch_size < data_len:
        ret_x, ret_len = data_x[epoch_size*batch_size:], len_list[epoch_size*batch_size:]
        yield (ret_x, ret_len) 

def data_iter_indices(data_len, batch_size):
    """
    Return iteration indices
    """
    epoch_size = data_len // batch_size
    
    for _ in range(epoch_size):
        indices = np.random.choice(data_len, batch_size)
        yield indices

def pred_data_iter_indices(data_len, batch_size): 
    """return indices"""
    epoch_size = data_len // batch_size
    
    for i in xrange(epoch_size):
        indices = np.arange(i*batch_size, (i+1)*batch_size)
        yield indices
    if epoch_size*batch_size < data_len:
        indices = np.arange(epoch_size*batch_size, data_len)
        yield indices

"""confusion calculation and logging"""
def calculate_confusion_single(pred_list, label_list, label_size):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros((label_size, label_size), dtype=np.int32)
    for i in xrange(len(label_list)):
        confusion[label_list[i], pred_list[i]] += 1
    
    tp_fp = np.sum(confusion, axis=0)
    tp_fn = np.sum(confusion, axis=1)
    tp = np.array([confusion[i, i] for i in range(len(confusion))])
    
    precision = tp.astype(np.float32)/(tp_fp+1e-40)
    recall = tp.astype(np.float32)/(tp_fn+1e-40)
    overall_prec = np.float(np.sum(tp))/(np.sum(tp_fp)+1e-40)
    overall_recall = np.float(np.sum(tp))/(np.sum(tp_fn)+1e-40)
    
    return precision, recall, overall_prec, overall_recall, confusion

def print_confusion_single(prec, recall, overall_prec, overall_recall, num_to_tag):
    """Helper method that prints confusion matrix."""
    logstr=""
    logstr += '{:15}\t{:7}\t{:7}\n'.format('TAG', 'Prec', 'Recall')
    for i, tag in sorted(num_to_tag.items()):
        logstr += '{:15}\t{:2.4f}\t{:2.4f}\n'.format(tag.encode('utf-8'), prec[i], recall[i])
    logstr += '{:15}\t{:2.4f}\t{:2.4f}\n'.format('OVERALL', overall_prec, overall_recall)
    logging.info(logstr)
    print logstr

def calculate_confusion_multi(pred_matrix, label_matrix):
    """Helper method that calculates confusion matrix."""
    tp = np.sum(np.logical_and(pred_matrix, label_matrix), axis=0)
    tp_fp = np.sum(pred_matrix, axis=0)
    tp_fn = np.sum(label_matrix, axis=0)
    precision = tp.astype(np.float32)/(tp_fp+1e-40)
    recall = tp.astype(np.float32)/(tp_fn+1e-40)
    overall_prec = np.float(np.sum(tp))/(np.sum(tp_fp)+1e-40)
    overall_recall = np.float(np.sum(tp))/(np.sum(tp_fn)+1e-40)
    return precision, recall, overall_prec, overall_recall

def print_confusion_multi(prec, recall, overall_prec, overall_recall, num_to_tag):
    """Helper method that prints confusion matrix."""
    logstr=""
    logstr += '{:15}\t{:7}\t{:7}\n'.format('TAG', 'Prec', 'Recall')
    for i, tag in sorted(num_to_tag.items()):
        logstr += '{:15}\t{:2.4f}\t{:2.4f}\n'.format(tag.encode('utf-8'), prec[i], recall[i])
    logstr += '{:15}\t{:2.4f}\t{:2.4f}\n'.format('OVERALL', overall_prec, overall_recall)
    #logging.info(logstr)
    print logstr
