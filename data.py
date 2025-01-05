import numpy as np
import torch
import pandas as pd
import sys, os, time

class AlleleData(torch.utils.data.Dataset):
    # for now, just use the homo/hetero-zygous information
    def __init__(self, file_prefix, root_dir):

        self.num_satellites = 16

        for region in ["savannah", "forest"]:
            df = pd.read_csv(file_prefix+f"{region}.txt", sep="\s|\t", header=None)
            df.columns = ["name", "zone", *range(self.num_satellites)]

            label_lookup = {}
            with open(f"{root_dir}{region}.txt", "r") as f:
                for long_line in f:
                    line = long_line.replace("\n", "").split(" ")
                    label_lookup[line[1]] = (line[3], line[4])
            
            lat_column, long_column = [], []
            for row, item in df.iterrows():
                lat_column.append(float(label_lookup[str(item["zone"])][0]))
                long_column.append(float(label_lookup[str(item["zone"])][1]))
            df["lat"] = lat_column
            df["long"] = long_column

            try:
                self.data.head()
                self.data = pd.concat([self.data, df])
            except:
                self.data = df
        
        self.data = self.data.sample(frac=1)
        self.cat_summaries = self.data.iloc[:, 2:18].nunique(axis=0).tolist()

        self.one_hot_enc_lookup = {} # key=col / microsatellite location, value=arr of unique vals
        for i in range(self.num_satellites):
            self.one_hot_enc_lookup[i] = self.data[i].unique()
        self.root_dir = root_dir

        self.max_width = max(self.cat_summaries)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        category = self.data.iloc[idx, 2:self.num_satellites+2]
        one_hot_encs = torch.zeros((self.num_satellites, self.max_width))
        for i in range(self.num_satellites):
            # inp = torch.zeros(max_width)
            local_cat_pos = np.where(self.one_hot_enc_lookup[i] == category[i])
            one_hot_encs[i, local_cat_pos] = 1
            # inp[local_cat_pos] = 1
            # one_hot_encs.append(inp)
        
        datum = {"genotype": one_hot_encs, "location": torch.tensor(self.data.iloc[idx, self.num_satellites+2:].tolist())}
        # print(type(one_hot_encs), type(self.data.iloc[idx, 18:].values))
        # return one_hot_encs, self.data.iloc[idx, 18:].values
        return datum


    def get_categorical_summaries(self):
        return self.data.iloc[:, 2:18].nunique(axis=0).tolist()
