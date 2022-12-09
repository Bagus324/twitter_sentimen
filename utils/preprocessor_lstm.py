import pandas as pd
import sys
import nltk
from tqdm import tqdm
import re
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import os
import pytorch_lightning as pl
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
class PreprocessorLSTM():
    def __init__(self,
                save_dir,
                d_binary,
                d_ite,
                batch_size=10
                ) -> None:
        self.token = Tokenizer()
        self.device = torch.device("cuda:0")
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.max_length = 142
        self.d_binary=d_binary
        self.d_ite=d_ite
        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')
    
    def main(self):
        if os.path.exists(self.save_dir):
            print("Loading Merged Dataset")
            self.load_binary = pd.read_pickle(self.save_dir)
            print("Load Complete")
            print(self.load_binary.head())
        else:
            print("Processing")
            self.load_binary = pd.read_csv(self.d_binary, encoding="ISO-8859-1")
            self.load_ite = pd.read_csv(self.d_ite, encoding="ISO-8859-1")
            self.converter()
            self.load_binary["Tweet"] = self.load_binary["Tweet"].apply(lambda x: f"{self.clean_sentence(x)}")
            self.load_binary.to_pickle(self.save_dir)
            self.load_binary.to_csv('datasets/merged_tweet.csv')
        self.data_binary_list = self.separating_label()
        x_pad, y = self.tokenizing(self.data_binary_list)
        vocab_size, embedding_size, weight = self.embedding(x_pad)
        x_train, x_val, y_train, y_val = self.arrange_data_w2v(x_pad, y)
        # print(x_train.shape)
        # print(x_val.shape)
        # print(y_train.shape)
        # print(y_val.shape)
        # print(vocab_size)
        # print(embedding_size)
        return x_train, x_val, y_train, y_val, vocab_size, embedding_size, weight
        # data_train, data_test, data_valid = self.arrange_data(self.data_binary_list)
        # return data_train, data_valid, data_test

    def tokenizing(self, datas):
        x = []
        y = []
        for line in tqdm(datas, desc="Getting Label"):
            x.append(line[0])
            # temp = line[0]
            # for word in temp:
            #     x.append(word_tokenize(word))
            #     print(word)
            #     sys.exit()
            y.append(line[1:])
        tokenizer = Tokenizer()
        fit_text = x
        tokenizer.fit_on_texts(fit_text)
        sequences = tokenizer.texts_to_sequences(x)
        list_set_sequence = [list(dict.fromkeys(seq)) for seq in sequences]

        print('Padding Data...')
        padded = pad_sequences([list(list_set_sequence[i]) for i in range(len(list_set_sequence))], maxlen=self.max_length, padding='post', truncating='post', value=0)
        print('[Padding Completed]\n')
        return padded ,y
    def embedding(self, x):
        print('Creating Embedding')
        x_vector = []
        if os.path.exists("model/w2v.model"):
            print("Loading w2v")
            w2v_model = Word2Vec.load("model/w2v.model")
        else:
            print("Traingin w2v")
            w2v_model = Word2Vec(vector_size=100,workers=2, min_count=1)
            w2v_model.build_vocab(x)
            w2v_model.train(x, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)
            w2v_model.save("model/w2v.model")
        vectors = w2v_model.wv
        w2v_weights = w2v_model.wv.vectors
        vocab_size, embedding_size = w2v_weights.shape
        max = 142
        print("Vectorising")
        for i, word in tqdm(enumerate(x)):
            temp = []
            for kata in word:
                temp.append(vectors[kata])
            x_vector.append(temp)

        return vocab_size, embedding_size, w2v_weights
        # x_train = torch.tensor(x_pad)
        # y = torch.tensor(y)
        # train_tensor_dataset, test_tensor_dataset = torch.utils.data.random_split(x_pad, [round(len(x) * 0.8), round(len(x) * 0.2)])

    def arrange_data_w2v(self, x, y):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        x_val = np.asarray(x_val)
        y_val = np.asarray(y_val)

        # x_train = torch.from_numpy(x_train)
        # y_train = torch.Tensor(y_train)

        # x_val = torch.from_numpy(x_val)
        # y_val = torch.Tensor(y_val)


        # x_train = x_train.to(self.device)
        # y_train = y_train.to(self.device)

        # x_val = x_val.to(self.device)
        # y_val = y_val.to(self.device)

        # train_data = TensorDataset(x_train, y_train)
        # val_data = TensorDataset(x_val, y_val)

        # train_loader = DataLoader(train_data, shuffle=True, batch_size = self.batch_size)
        # val_loader = DataLoader(val_data, shuffle=True, batch_size = self.batch_size)

        return x_train, x_val, y_train, y_val
    def converter(self):
        print('Converting and Merging Data')
        x=1
        for i, sentimen in enumerate(self.load_ite["sentimen"]):
            if sentimen == 0:
                if x == 1:
                    list_1 = [self.load_ite["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,0,0,0]
                    x+=1
                elif x == 2:
                    list_2 = [self.load_ite["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,0,0,0]
                    a = pd.DataFrame([[i for i in list_1], [i for i in list_2]],
                                        columns=["Tweet",
                                                "HS",
                                                "Abusive",
                                                "HS_Individual",
                                                "HS_Group",
                                                "HS_Religion",
                                                "HS_Race",
                                                "HS_Physical",
                                                "HS_Gender",
                                                "HS_Other",
                                                "HS_Weak",
                                                "HS_Moderate",
                                                "HS_Strong"])
                    self.load_binary = pd.concat([self.load_binary, a], ignore_index=True)
                    x=1
            elif sentimen == 1:
                if x == 1:
                    list_1 = [self.load_ite["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,0,0,0]
                    x+=1
                elif x == 2:
                    list_2 = [self.load_ite["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,0,0,0]
                    a = pd.DataFrame([[i for i in list_1], [i for i in list_2]],
                                        columns=["Tweet",
                                                "HS",
                                                "Abusive",
                                                "HS_Individual",
                                                "HS_Group",
                                                "HS_Religion",
                                                "HS_Race",
                                                "HS_Physical",
                                                "HS_Gender",
                                                "HS_Other",
                                                "HS_Weak",
                                                "HS_Moderate",
                                                "HS_Strong"])
                    self.load_binary = pd.concat([self.load_binary, a], ignore_index=True)
                    x=1
#===============================================================================================
            elif sentimen == 1:
                if x == 1:
                    list_1 = [self.load_ite["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,0,0,0]
                    x+=1
                elif x == 2:
                    list_2 = [self.load_ite["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,0,0,0]
                    a = pd.DataFrame([[i for i in list_1], [i for i in list_2]],
                                        columns=["Tweet",
                                                "HS",
                                                "Abusive",
                                                "HS_Individual",
                                                "HS_Group",
                                                "HS_Religion",
                                                "HS_Race",
                                                "HS_Physical",
                                                "HS_Gender",
                                                "HS_Other",
                                                "HS_Weak",
                                                "HS_Moderate",
                                                "HS_Strong"])
                    self.load_binary = pd.concat([self.load_binary, a], ignore_index=True)
                    x=1
#===============================================================================================
            elif sentimen == 2:
                if x == 1:
                    list_1 = [self.load_ite["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,1,0,0]
                    x+=1
                elif x == 2:
                    list_2 = [self.load_ite["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,1,0,0]
                    a = pd.DataFrame([[i for i in list_1], [i for i in list_2]],
                                        columns=["Tweet",
                                                "HS",
                                                "Abusive",
                                                "HS_Individual",
                                                "HS_Group",
                                                "HS_Religion",
                                                "HS_Race",
                                                "HS_Physical",
                                                "HS_Gender",
                                                "HS_Other",
                                                "HS_Weak",
                                                "HS_Moderate",
                                                "HS_Strong"])
                    self.load_binary = pd.concat([self.load_binary, a], ignore_index=True)
                    x=1
#===============================================================================================
            elif sentimen == 4:
                if x == 1:
                    list_1 = [self.load_ite["Tweet"].loc[i],1,1,1,0,0,0,1,1,0,0,1,0]
                    x+=1
                elif x == 2:
                    list_2 = [self.load_ite["Tweet"].loc[i],1,1,1,0,0,0,1,1,0,0,1,0]
                    a = pd.DataFrame([[i for i in list_1], [i for i in list_2]],
                                        columns=["Tweet",
                                                "HS",
                                                "Abusive",
                                                "HS_Individual",
                                                "HS_Group",
                                                "HS_Religion",
                                                "HS_Race",
                                                "HS_Physical",
                                                "HS_Gender",
                                                "HS_Other",
                                                "HS_Weak",
                                                "HS_Moderate",
                                                "HS_Strong"])
                    self.load_binary = pd.concat([self.load_binary, a], ignore_index=True)
                    x=1
#===============================================================================================
            elif sentimen == 5:
                if x == 1:
                    list_1 = [self.load_ite["Tweet"].loc[i],1,1,1,0,0,0,0,1,0,0,1,0]
                    x+=1
                elif x == 2:
                    list_2 = [self.load_ite["Tweet"].loc[i],1,1,1,0,0,0,0,1,0,0,1,0]
                    a = pd.DataFrame([[i for i in list_1], [i for i in list_2]],
                                        columns=["Tweet",
                                                "HS",
                                                "Abusive",
                                                "HS_Individual",
                                                "HS_Group",
                                                "HS_Religion",
                                                "HS_Race",
                                                "HS_Physical",
                                                "HS_Gender",
                                                "HS_Other",
                                                "HS_Weak",
                                                "HS_Moderate",
                                                "HS_Strong"])
                    self.load_binary = pd.concat([self.load_binary, a], ignore_index=True)
                    x=1
#===============================================================================================
            elif sentimen == 6:
                if x == 1:
                    list_1 = [self.load_ite["Tweet"].loc[i],1,1,0,1,1,1,1,0,1,0,0,1]
                    x+=1
                elif x == 2:
                    list_2 = [self.load_ite["Tweet"].loc[i],1,1,0,1,1,1,1,0,1,0,0,1]
                    a = pd.DataFrame([[i for i in list_1], [i for i in list_2]],
                                        columns=["Tweet",
                                                "HS",
                                                "Abusive",
                                                "HS_Individual",
                                                "HS_Group",
                                                "HS_Religion",
                                                "HS_Race",
                                                "HS_Physical",
                                                "HS_Gender",
                                                "HS_Other",
                                                "HS_Weak",
                                                "HS_Moderate",
                                                "HS_Strong"])
                    self.load_binary = pd.concat([self.load_binary, a], ignore_index=True)
                    x=1
        print('Data Merged')
        return self.load_binary

    def clean_sentence(self, sentence):
        # Membersihkan dari karakter tidak standard
        sentence = re.sub(r"[^A-Za-z(),!?\'\`]", " ", sentence)

        sentence = re.sub(r"\'s", " \'s", sentence)
        sentence = re.sub(r"\'ve", " \'ve", sentence)
        sentence = re.sub(r"n\'t", " n\'t", sentence)
        sentence = re.sub(r"\n", "", sentence)
        sentence = re.sub(r"\'re", " \'re", sentence)
        sentence = re.sub(r"\'d", " \'d", sentence)
        sentence = re.sub(r"\'ll", " \'ll", sentence)
        sentence = re.sub(r",", " , ", sentence)
        sentence = re.sub(r"!", " ! ", sentence)
        sentence = re.sub(r"'", "", sentence)
        sentence = re.sub(r'""', "", sentence)
        sentence = re.sub(r"\(", "", sentence)
        sentence = re.sub(r"\)", "", sentence)
        sentence = re.sub(r"\?", " \? ", sentence)
        sentence = re.sub(r"\,", "", sentence)
        sentence = re.sub(r"\s{2,}", " ", sentence)
        sentence = sentence.lower()
        sentence = re.sub(r"user", "", sentence)

        return sentence.strip()

    
    # def arrange_data(self, datas):
    #     x_input_ids, x_token_type_ids, x_attention_mask, y = [], [], [], []

    #     for i, tr_d in enumerate(datas):
    #         title = tr_d[0]
    #         label = tr_d[1:]
    #         binary_lbl = label
    #         # binary_lbl[label] = 1
            
    #         tkn = self.tokenizer(text = title, 
    #                             max_length= self.max_length, 
    #                             padding='max_length',
    #                             truncation=True)
            
            
    #         x_input_ids.append(tkn['input_ids'])
    #         x_token_type_ids.append(tkn['token_type_ids'])
    #         x_attention_mask.append(tkn['attention_mask'])
    #         y.append(binary_lbl)
    #         # if i > 100: break


    #     x_input_ids = torch.tensor(x_input_ids)

    #     x_token_type_ids = torch.tensor(x_token_type_ids)
    #     x_attention_mask = torch.tensor(x_attention_mask)
    #     y = torch.tensor(y)
    #     train_valid_length = len(x_input_ids)* 0.8
    #     tensor_dataset = TensorDataset(x_input_ids, x_token_type_ids, x_attention_mask, y)
    #     train_tensor_dataset, test_tensor_dataset = torch.utils.data.random_split(tensor_dataset, [round(len(x_input_ids) * 0.8), round(len(x_input_ids) * 0.2)])
    #     train_tensor_dataset, valid_tensor_dataset = torch.utils.data.random_split(train_tensor_dataset, [round(train_valid_length * 0.9), round(train_valid_length * 0.1)])
    #     if not os.path.exists(f"preprocessed/train.pt") and not os.path.exists(f"preprocessed/test.pt"):
    #         torch.save(train_tensor_dataset, f"preprocessed/train.pt")
    #         torch.save(test_tensor_dataset, f"preprocessed/test.pt")
    #         torch.save(valid_tensor_dataset, f"preprocessed/valid.pt")
    #     return train_tensor_dataset, test_tensor_dataset, valid_tensor_dataset

        
    def separating_label(self):
        print('Separating Labels')
        final_data = []
        for line in tqdm(self.load_binary.values.tolist()):
            label = line[1:]
            indexing = [i for i, l in enumerate(label) if l == 1]
            if len(indexing) >= 1:
                for i, isi in enumerate(label):
                    wrapper = np.zeros((12), dtype=int).tolist()
                    if isi == 1:
                        wrapper[i]=1
                        final_data.append([line[0]]+wrapper)
        print('Separated')
        return final_data

    # def setup(self, stage = None):
    #     train_data, valid_data, test_data = self.main()
    #     if stage == "fit":
    #         self.train_data = train_data
    #         self.valid_data = valid_data
    #     elif stage == "predict":
    #         self.test_data = test_data

    # def train_dataloader(self):
    #     sampler = RandomSampler(self.train_data)
    #     return DataLoader(
    #         dataset = self.train_data,
    #         batch_size = self.batch_size,
    #         sampler = sampler,
    #         num_workers = 1
    #     )

    # def val_dataloader(self):
    #     sampler = RandomSampler(self.valid_data)
    #     return DataLoader(
    #         dataset = self.valid_data,
    #         batch_size = self.batch_size,
    #         sampler = sampler,
    #         num_workers = 1
    #     )

    # def predict_dataloader(self):
    #     sampler = SequentialSampler(self.test_data)
    #     return DataLoader(
    #         dataset = self.test_data,
    #         batch_size = self.batch_size,
    #         sampler = sampler,
    #         num_workers = 1
    #     )


    def preprocessing_glove(self):
        if os.path.exists(self.save_dir):
            print("Loading Merged Dataset")
            self.load_binary = pd.read_pickle(self.save_dir)
            print("Load Complete")
            print(self.load_binary.head())
        else:
            print("Processing")
            self.load_binary = pd.read_csv(self.d_binary, encoding="ISO-8859-1")
            self.load_ite = pd.read_csv(self.d_ite, encoding="ISO-8859-1")
            self.converter()
            self.load_binary["Tweet"] = self.load_binary["Tweet"].apply(lambda x: f"{self.clean_sentence(x)}")
            self.load_binary.to_pickle(self.save_dir)
            self.load_binary.to_csv('datasets/merged_tweet.csv')
        self.data_binary_list = self.separating_label()
        vocab_size, x_train, x_val, y_train, y_val = self.arrange_data_glove(self.data_binary_list)
        weight = self.vectorizing_glove(vocab_size)
        return vocab_size, weight, x_train, x_val, y_train, y_val

    def arrange_data_glove(self, datas):
        x = []
        y = []
        for line in tqdm(datas, desc="Getting Label"):
                x.append(line[0])
                # temp = line[0]
                # for word in temp:
                #     x.append(word_tokenize(word))
                #     print(word)
                #     sys.exit()
                y.append(line[1:])
        self.token.fit_on_texts(x)
        seq = self.token.texts_to_sequences(x)
        padded_seq = pad_sequences(seq, maxlen=142, padding="post", truncating="post", value=0)
        vocab_size = len(self.token.word_index)+1

        x_train, x_val, y_train, y_val = train_test_split(padded_seq, y, test_size=0.2, random_state=42)


        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        x_val = np.asarray(x_val)
        y_val = np.asarray(y_val)
        return vocab_size, x_train, x_val, y_train, y_val

    def vectorizing_glove(self, vocab_size):
        embedding_vector = {}
        f = open('utils/glove_50dim_wiki.id.case.text.txt', encoding="ISO-8859-1")
        for line in tqdm(f, desc='Creating Embedding Words'):
            value = line.split(' ')
            word = value[0]
            coef = np.array(value[1:],dtype = 'float32')
            embedding_vector[word] = coef
        
        embedding_matrix = np.zeros((vocab_size,50))
        for word,i in tqdm(self.token.word_index.items(), desc='Bikin Matrix'):
            embedding_value = embedding_vector.get(word)
            if embedding_value is not None:
                embedding_matrix[i] = embedding_value
        

        return embedding_matrix

            