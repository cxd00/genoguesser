import numpy as np
import torch
import pandas as pd
import sys, os, time

class AlleleData(torch.utils.data.Dataset):
    # for now, just use the homo/hetero-zygous information
    def __init__(self, file_prefix, root_dir):
        for region in ["savannah", "forest"]:
            df = pd.read_csv(file_prefix+f"{region}.txt", sep="\s|\t", header=None)
            df.columns = ["name", "zone", *range(16)]

            label_lookup = {}
            with open(f"{root_dir}{region}.txt", "r") as f:
                for long_line in f:
                    line = long_line.replace("\n", "").split(" ")
                    print(line)
                    label_lookup[line[1]] = (line[3], line[4])
            
            lat_column, long_column = [], []
            for row, item in df.iterrows():
                lat_column.append(label_lookup[str(item["zone"])][0])
                long_column.append(label_lookup[str(item["zone"])][1])
            df["lat"] = lat_column
            df["long"] = long_column

            try:
                self.data.head()
                self.data = pd.concat([self.data, df])
            except:
                self.data = df
        
        self.cat_summaries = self.data.iloc[:, 2:18].nunique(axis=0).tolist()

        self.one_hot_enc_lookup = {} # key=col / microsatellite location, value=arr of unique vals
        for i in range(16):
            self.one_hot_enc_lookup[i] = self.data[i].unique()
        print(self.data.head())
        self.root_dir = root_dir


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        one_hot_encs = []
        category = self.data.iloc[idx, 2:18]
        for i in range(16):
            inp = torch.zeros(self.cat_summaries[i])
            local_cat_pos = np.where(self.one_hot_enc_lookup[i] == category[i])
            inp[local_cat_pos] = 1
            one_hot_encs.append(inp)
        print(one_hot_encs)
        
        datum = {"genotype": one_hot_encs, "location":self.data.iloc[idx, 18:].values}
        return datum


    def get_categorical_summaries(self):
        return self.data.iloc[:, 2:18].nunique(axis=0).tolist()
