import numpy as np
import torch
import pandas as pd
import sys, os, time

class AlleleData(torch.utils.data.Dataset):
    # for now, just use the homo/hetero-zygous information
    def __init__(self, file_prefix, root_dir, seed=0):

        self.num_satellites = 16
        lat_column, long_column = [], []

        for region in ["savannah", "forest"]:
            df = pd.read_csv(file_prefix+f"{region}.txt", sep="\s|\t", header=None)
            df.columns = ["name", "zone", *range(self.num_satellites)]

            label_lookup = {}
            with open(f"{root_dir}{region}.txt", "r") as f:
                for long_line in f:
                    line = long_line.replace("\n", "").split(" ")
                    label_lookup[line[1]] = (line[3], line[4])
            
            for row, item in df.iterrows():
                lat_column.append(float(label_lookup[str(item["zone"])][0]))
                long_column.append(float(label_lookup[str(item["zone"])][1]))
            
            if hasattr(self, "data"):
                print("adding forest...")
                self.data = pd.concat([self.data, df])
                print(len(self.data))
            else:
                print("adding savannah...")
                self.data = df

        # clean up target data
        self.data["lat"], self.lat_min, self.lat_max = self.scale(lat_column)
        self.data["long"], self.long_min, self.long_max = self.scale(long_column)
        self.data["orig_lat"] = lat_column
        self.data["orig_long"] = long_column
        
        self.data = self.data.sample(frac=1, random_state=int(seed)).reset_index(drop=True) # shuffled

        cat_medians = self.get_categorical_medians() # produces median value of each column
        for idx, col in enumerate(self.data.columns[2:-4]): # iterates over all microsat data and replaces nans w/ the median
            self.data.loc[self.data[col] == -999, col] = cat_medians[idx]
        
        self.cat_summaries = self.data.iloc[:, 2:18].nunique(axis=0).tolist()

        self.one_hot_enc_lookup = {} # key=col / microsatellite location, value=arr of unique vals
        for i in range(self.num_satellites):
            print(self.data.columns[i])
            self.one_hot_enc_lookup[i] = torch.from_numpy(self.data[i].unique())
        self.root_dir = root_dir

        self.max_width = max(self.cat_summaries)


    def scale(self, lat_column):
        # return lat_column, 0, 0
        lat_column = np.array(lat_column)
        # lat_mean, lat_std = np.nanmean(lat_column), np.nanstd(lat_column)
        # lat_norm = (np.array(lat_column) - lat_mean) / lat_std
        # return lat_norm, lat_mean, lat_std 

        lat_min, lat_max = np.nanmin(lat_column), np.nanmax(lat_column)
        return (lat_column - lat_min) / (lat_max - lat_min), lat_min, lat_max


    def unscale(self, coords):
        # return coords, 0, 0
        lat_column, long_column = coords[:, 0], coords[:, 1]
        
        # return torch.cat([lat_column * self.lat_max + self.lat_min, \
        #         long_column * self.long_max + self.long_min]).view(lat_column.shape[0], 2)
        return torch.stack((lat_column * (self.lat_max - self.lat_min) + self.lat_min, \
                long_column * (self.long_max - self.long_min) + self.long_min), dim=1)
        

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        category = torch.from_numpy(self.data.iloc[idx, 2:self.num_satellites+2].astype(int).values)
        ordinals = torch.zeros(self.num_satellites)
        for i in range(self.num_satellites):
            ordinal_idx = torch.where(self.one_hot_enc_lookup[i] == category[i])
            ordinals[i] = ordinal_idx[0]
        datum = {"genotype": ordinals.long(), "location": torch.tensor(self.data.iloc[idx, self.num_satellites+2:-2].tolist())}
        return datum


    def __oldgetitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        category = torch.from_numpy(self.data.iloc[idx, 2:self.num_satellites+2].astype(int).values)
        one_hot_encs = torch.zeros((self.num_satellites, self.max_width))
        for i in range(self.num_satellites):
            # inp = torch.zeros(max_width)
            local_cat_pos = torch.where(self.one_hot_enc_lookup[i] == category[i])
            one_hot_encs[i, local_cat_pos] = 1.
            # inp[local_cat_pos] = 1
            # one_hot_encs.append(inp)
        
        datum = {"genotype": one_hot_encs, "location": torch.tensor(self.data.iloc[idx, self.num_satellites+2:-2].tolist())}
        # print(type(one_hot_encs), type(self.data.iloc[idx, 18:].values))
        # return one_hot_encs, self.data.iloc[idx, 18:].values
        return datum


    # def replace_missing(self, data):
    #     for idx, row in data.iterrows():
    #         if 
    #     return data


    def get_categorical_summaries(self):
        return self.data.iloc[:, 2:18].nunique(axis=0).tolist()


    def get_categorical_medians(self):
        return self.data.iloc[:, 2:18].median().tolist()
