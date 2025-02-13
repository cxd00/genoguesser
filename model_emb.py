import torch
import lightning
import torchvision
import os, sys, time
import numpy as np

class GenoLoc(torch.nn.Module):
    def __init__(self, microsat_summary):
        super().__init__()
        self.embedding_summary = microsat_summary
        self.embedding_layers = [] 
        self.total_emb_features = 0
        
        for unique_cats in self.embedding_summary:
            output_dim = int(np.sqrt(int(unique_cats)))
            self.total_emb_features += output_dim
            emb = torch.nn.Embedding(unique_cats, output_dim)
            self.embedding_layers.append(emb)
            # self.reshape_layers.append(torch.nn.Flatten())

        self.bn = torch.nn.BatchNorm1d(self.total_emb_features)
        self.ln = []
        self.ln.append(torch.nn.Linear(self.total_emb_features, 256))
        # self.ln.append(torch.nn.BatchNorm1d(256))
        for i in range(2):
            self.ln.append(torch.nn.Linear(256, 256))
            self.ln.append(torch.nn.ELU())
        self.ln.append(torch.nn.Dropout(p=0.3))
        # self.ln.append(torch.nn.BatchNorm1d(256))
        for i in range(3):
            self.ln.append(torch.nn.Linear(256, 256))
            self.ln.append(torch.nn.ELU())
        self.ln.append(torch.nn.Dropout(p=0.3))

        self.ln.append(torch.nn.Linear(256, 2))
        self.ln.append(torch.nn.Linear(2, 2))

        self.embedding_layers = torch.nn.ModuleList(self.embedding_layers)
        self.ln = torch.nn.ModuleList(self.ln)

        # self.learning_rate = 0.01
        # self.weight_decay = 0.0005


    def forward(self, x):
        x = [emb(x[:,idx]) for idx, emb in enumerate(self.embedding_layers)]
        x = torch.cat(x, dim=1)
        x = self.bn(x)
        for lyr in self.ln:
            x = lyr(x)
        return x

    
    """
    def training_step(self, batch, batch_idx):
        inputs = batch["genotype"]
        targets = batch["location"]
        outputs = self(inputs)
        loss = self.euclidean_distance_loss(outputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        if batch_idx % 50 == 0:
            print(loss)
        return loss

    
    def configure_optimizers(self):
        param_dicts = [{
            "params": [p for n, p in self.named_parameters()],
            "lr": self.learning_rate,
        }]
        
        return torch.optim.AdamW(
            param_dicts, lr=self.learning_rate, weight_decay=self.weight_decay
        )



    def validation_step(self, batch, batch_idx):
        inputs = batch["genotype"]
        targets = batch["location"]
        outputs = self(inputs)
        loss = self.euclidean_distance_loss(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)


    """
    def euclidean_distance_loss(self, y_true, y_pred):
        return torch.sqrt((y_true - y_pred)**2).sum()


    def haversine_distance(self, pred, target, epsSq = 1.e-7, epsAs = 1.e-5):   # add optional epsilons to avoid singularities
        # print ('haversine_loss: epsSq:', epsSq, ', epsAs:', epsAs)
        lat1, lon1 = torch.split(pred, 1, dim=1)
        lat2, lon2 = torch.split(target, 1, dim=1)
        r = 6371  # Radius of Earth in kilometers
        phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
        delta_phi, delta_lambda = torch.deg2rad(lat2-lat1), torch.deg2rad(lon2-lon1)
        a = torch.sin(delta_phi/2)**2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda/2)**2
        # return tensor.mean(2 * r * torch.asin(torch.sqrt(a)))
        # "+ (1.0 - a**2) * epsSq" to keep sqrt() away from zero
        # "(1.0 - epsAs) *" to keep asin() away from plus-or-minus one
        return 2 * r * torch.asin ((1.0 - epsAs) * torch.sqrt (a + (1.0 - a**2) * epsSq)) # as kms

    def haversine_loss(self, pred, target):
        dist = self.haversine_distance(pred, target)
        return torch.Tensor.mean(dist)

