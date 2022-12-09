import numpy as np

from tqdm import tqdm


class Sequencer():

    def __init__(self,
                 all_words,
                 max_words,
                 seq_len,
                 embedding_matrix):

        self.seq_len = seq_len
        self.embed_matrix = embedding_matrix
        temp_vocab = list(set(all_words))
        self.vocab = []
        self.word_cnts = {}

        for word in tqdm(temp_vocab, desc='Step 2'):
            count = len([0 for w in all_words if w == word])
            self.word_cnts[word] = count
            counts = list(self.word_cnts.values())
            indexes = list(range(len(counts)))

        cnt = 0
        while cnt + 1 != len(counts):
            cnt = 0
            for i in range(len(counts)-1):
                if counts[i] < counts[i+1]:
                    counts[i+1], counts[i] = counts[i], counts[i+1]
                    indexes[i], indexes[i+1] = indexes[i+1], indexes[i]
                else:
                    cnt += 1

        for ind in tqdm(indexes[:max_words], desc='Step 3'):
            self.vocab.append(temp_vocab[ind])

    def text_to_vector(self, text):
        tokens = text.split()
        len_v = len(tokens)-1 if len(tokens) < self.seq_len else self.seq_len-1
        vec = []
        for tok in tokens[:len_v]:
            try:
                vec.append(self.embed_matrix[tok])
            except Exception:
                pass

        last_pieces = self.seq_len - len(vec)
        for i in range(last_pieces):
            vec.append(np.zeros(300,))

        return np.asarray(vec).flatten()