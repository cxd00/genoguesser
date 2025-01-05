from model import GenoLoc
from data import AlleleData
from lightning import Trainer

import torch

DEVICE = "cuda"

def train(data, model):
    batch_size = 8
    num_epochs = 100
    
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())

    train, test, val = torch.utils.data.random_split(data, [0.7, 0.15, 0.15])
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True) 
    valloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for batch_idx, datum in enumerate(trainloader):
            optimizer.zero_grad()
            geno, loc = datum["genotype"].to(DEVICE), datum["location"].to(DEVICE)
            output = model(geno)
            loss = model.euclidean_distance_loss(output, loc)
            loss.backward()
            optimizer.step()
        
        # val_output = model()
        # val_loss = model.euclidean_distance_loss()
        record = f"Epoch {epoch}: training - {loss}, val - {val_loss}"


if __name__ == "__main__":
    data = AlleleData("~/uw/cefs/REFELE_5.8_filtered_", "/home/cynthia/uw/cefs/zones_44_")
    train, test, val = torch.utils.data.random_split(data, [0.7, 0.15, 0.15])
    batch_size = 32
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True) 
    valloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)

    model = GenoLoc(data.cat_summaries)

    trainer = Trainer(max_epochs=100)
    trainer.fit(model=model, train_dataloaders=trainloader) 
    
