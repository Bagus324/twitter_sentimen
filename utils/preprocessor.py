import pandas as pd
import sys
from tqdm import tqdm
import re
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import os
import pytorch_lightning as pl
class Preprocessor(pl.LightningDataModule):
    def __init__(self,
                d_binary="datasets/binary.csv",
                d_ite="datasets/Dataset Twitter Fix - Indonesian Sentiment Twitter Dataset Labeled (1).csv",
                max_length=100,
                batch_size=8) -> None:
        self.batch_size = batch_size
        self.max_length = max_length
        self.d_binary=d_binary
        self.d_ite=d_ite
        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')
    
    def main(self):
        self.load_binary = pd.read_csv(self.d_binary, encoding="ISO-8859-1")
        self.load_ite = pd.read_csv(self.d_ite, encoding="ISO-8859-1")
        condition_empty_label = self.load_binary[
            (
                (self.load_binary['HS'] == 0) &
                (self.load_binary['Abusive'] == 0) &
                (self.load_binary['HS_Individual'] == 0) &
                (self.load_binary['HS_Group'] == 0) &
                (self.load_binary['HS_Religion'] == 0) &
                (self.load_binary['HS_Race'] == 0) &
                (self.load_binary['HS_Physical'] == 0) &
                (self.load_binary['HS_Gender'] == 0) &
                (self.load_binary['HS_Other'] == 0) &
                (self.load_binary['HS_Weak'] == 0) &
                (self.load_binary['HS_Moderate'] == 0) &
                (self.load_binary['HS_Strong'] == 0)
            )
        ].index
        self.load_binary = self.load_binary.drop(condition_empty_label)
        self.converter()
        self.load_binary["Tweet"] = self.load_binary["Tweet"].apply(lambda x: f"{self.clean_sentence(x)}")
        self.data_binary_list = self.sepping()
        data_train, data_test, data_valid = self.arrange_data(self.data_binary_list)
        return data_train, data_valid, data_test
        


    def converter(self):
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

    
    def arrange_data(self, datas):
        x_input_ids, x_token_type_ids, x_attention_mask, y = [], [], [], []

        for i, tr_d in enumerate(datas):
            title = tr_d[0]
            label = tr_d[1:]
            binary_lbl = label
            # binary_lbl[label] = 1
            
            tkn = self.tokenizer(text = title, 
                                max_length= self.max_length, 
                                padding='max_length',
                                truncation=True)
            
            
            x_input_ids.append(tkn['input_ids'])
            x_token_type_ids.append(tkn['token_type_ids'])
            x_attention_mask.append(tkn['attention_mask'])
            y.append(binary_lbl)
            # if i > 100: break


        x_input_ids = torch.tensor(x_input_ids)

        x_token_type_ids = torch.tensor(x_token_type_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        y = torch.tensor(y)
        train_valid_length = len(x_input_ids)* 0.8
        tensor_dataset = TensorDataset(x_input_ids, x_token_type_ids, x_attention_mask, y)
        train_tensor_dataset, test_tensor_dataset = torch.utils.data.random_split(tensor_dataset, [round(len(x_input_ids) * 0.8), round(len(x_input_ids) * 0.2)])
        train_tensor_dataset, valid_tensor_dataset = torch.utils.data.random_split(train_tensor_dataset, [round(train_valid_length * 0.9), round(train_valid_length * 0.1)])
        if not os.path.exists(f"preprocessed/train.pt") and not os.path.exists(f"preprocessed/test.pt"):
            torch.save(train_tensor_dataset, f"preprocessed/train.pt")
            torch.save(test_tensor_dataset, f"preprocessed/test.pt")
            torch.save(valid_tensor_dataset, f"preprocessed/valid.pt")
        return train_tensor_dataset, test_tensor_dataset, valid_tensor_dataset

        
    def sepping(self):
        
        final_data = []
        for line in self.load_binary.values.tolist():
            label = line[1:]
            indexing = [i for i, l in enumerate(label) if l == 1]
            if len(indexing) >= 1:
                for i, isi in enumerate(label):
                    wrapper = np.zeros((12), dtype=int).tolist()
                    if isi == 1:
                        wrapper[i]=1
                        final_data.append([line[0]]+wrapper)
        
        return final_data

    def setup(self, stage = None):
        train_data, valid_data, test_data = self.main()
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "predict":
            self.test_data = test_data

    def train_dataloader(self):
        sampler = RandomSampler(self.train_data)
        return DataLoader(
            dataset = self.train_data,
            batch_size = self.batch_size,
            sampler = sampler,
            num_workers = 1
        )

    def val_dataloader(self):
        sampler = RandomSampler(self.valid_data)
        return DataLoader(
            dataset = self.valid_data,
            batch_size = self.batch_size,
            sampler = sampler,
            num_workers = 1
        )

    def predict_dataloader(self):
        sampler = SequentialSampler(self.test_data)
        return DataLoader(
            dataset = self.test_data,
            batch_size = self.batch_size,
            sampler = sampler,
            num_workers = 1
        )
    
