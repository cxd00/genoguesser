from genoguesser.model import GenoLoc
from genoguesser.data import AlleleData

import torch

DEVICE = "gpu"

def train(data, model):
    batch_size = 16
    num_epochs = 100
    
    model.to(DEVICE)
    data.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())

    train, test, val = torch.utils.data.random_split(data, [0.7, 0.15, 0.15])
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True) 
    valloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for batch_idx, datum in enumerate(trainloader):
            optimizer.zero_grad()
            output = model(datum["genotype"])
            loss = model.euclidean_distance_loss(output, datum["location"])
            loss.backward()
            optimizer.step()
        
        val_output = model(
        val_loss = model.euclidean_distance_loss()
        record = f"Epoch {epoch}: training - {loss}, val - {val_loss}"


if __name__ == "__main__":
    data = AlleleData("~/cefs/REFELE_5.8_filtered_", "cefs/zones_44_")
    model = GenoLoc(data.cat_summaries)
    
