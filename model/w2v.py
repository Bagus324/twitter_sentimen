import io
import os
import gensim
import multiprocessing

from tqdm import tqdm
from gensim.models import word2vec


class W2V():
    def __init__(self,
                 data,
                 vector_size,
                 corpus_txt='datasets/review_corpus.txt',
                 w2v_model_path='model/review_word2vec_model.model') -> None:

        self.data = data
        self.corpus_txt = corpus_txt
        self.w2v_model_path = w2v_model_path
        self.vector_size = vector_size

        if os.path.exists(self.w2v_model_path):
            print('Loading Word2Vec Model...')
            self.model = word2vec.Word2Vec.load(w2v_model_path)
            print('[ Loading Completed ]\n')
        elif os.path.exists(self.corpus_txt):
            self.train()
        else:
            self.create_corpus(self.data)
            self.train()

    def create_corpus(self, data):
        print('Creating Corpus...')
        with io.open(self.corpus_txt, 'w', encoding='utf-8') as corpus:
            for text in tqdm(data, desc='Creating Corpus'):
                corpus.write(" ".join(text) + '\n')
        print('[ Corpus Completed ]\n')

    def train(self):
        print('Training Word2Vec Model...')
        sentences = word2vec.LineSentence(self.corpus_txt)
        id_w2v = word2vec.Word2Vec(sentences, vector_size=self.vector_size, workers=multiprocessing.cpu_count()-1)
        id_w2v.save(self.w2v_model_path)
        self.model = id_w2v
        print('[ Training Completed ]\n')

    def w2v(self):
        return self.model

    def word2token(self, word):
        try:
            return self.model.wv.key_to_index[word]
        except KeyError:
            return 0

    def token2word(self, token):
        return self.model.wv.index_to_key[token]