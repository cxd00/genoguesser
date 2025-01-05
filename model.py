import torch
import lightning
import torchvision
import os, sys, time
import numpy as np

class GenoLoc(lightning.LightningModule):
    def __init__(self, microsat_summary):
        super().__init__()
        self.embedding_summary = microsat_summary
        self.embedding_layers = [] 
        self.total_emb_features = 0
        
        for unique_cats in self.embedding_summary:
            output_dim = int(np.sqrt(int(unique_cats)))
            self.total_emb_features += output_dim
            emb = torch.nn.Linear(unique_cats, output_dim)
            self.embedding_layers.append(emb)
            # self.reshape_layers.append(torch.nn.Flatten())

        # self.bn = torch.nn.BatchNorm1D(total_emb_features)(model)
        self.ln = []
        self.ln.append(torch.nn.Linear(self.total_emb_features, 256))
        for i in range(4):
            self.ln.append(torch.nn.Linear(256, 256))
            self.ln.append(torch.nn.ELU())
        self.ln.append(torch.nn.Dropout(p=0.25))
        for i in range(5):
            self.ln.append(torch.nn.Linear(256, 256))
            self.ln.append(torch.nn.ELU())

        self.ln.append(torch.nn.Linear(256, 2))
        self.ln.append(torch.nn.Linear(2, 2))

        self.embedding_layers = torch.nn.ModuleList(self.embedding_layers)
        self.ln = torch.nn.ModuleList(self.ln)


    def forward(self, x):
        emb_outputs = []
        for idx, emb in enumerate(self.embedding_layers):
            vec = x[:, idx, :self.embedding_summary[idx]]
            emb_outputs.append(emb(vec))

        x = torch.cat(emb_outputs, dim=1)
        for lyr in self.ln:
            x = lyr(x)
        return x

    
    def training_step(self, batch, batch_idx):
        inputs = batch["genotype"]
        targets = batch["location"]
        outputs = self(inputs)
        loss = self.euclidean_distance_loss(outputs, targets)
        if batch_idx == 20:
            print(loss)
        return loss

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.euclidean_distance_loss(outputs, targets)
        self.log("validation", loss)


    def euclidean_distance_loss(self, y_true, y_pred):
        return torch.sqrt((y_true - y_pred)**2).sum()



