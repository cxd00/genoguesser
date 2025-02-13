from model_emb import GenoLoc
from data import AlleleData

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping

import torch
from tqdm import tqdm
import wandb
import numpy as np
import sys
import time

DEVICE = "cuda"

logger = sys.argv[5]
form = sys.argv[4]
lr = float(sys.argv[3])
num_epochs = int(sys.argv[2])
batch_size = int(sys.argv[1])

def train(trainloader, valloader, model):
    
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    if logger == "wandb":
        run = wandb.init(project="genoguesser", config={
            "arch": "embeds",
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "step": 20
        })

    pbar = tqdm(total=num_epochs)
    best_val_loss = 1000
    for epoch in tqdm(range(num_epochs)):
        train_losses = []
        model.train()
        for batch_idx, datum in enumerate(trainloader):
            optimizer.zero_grad()
            geno, loc = datum["genotype"].to(DEVICE), datum["location"].to(DEVICE)
            output = model(geno)
            # loss = model.euclidean_distance_loss(data.unscale(output), data.unscale(loc))
            loss = model.haversine_loss(data.unscale(output), data.unscale(loc))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().numpy())
        pbar.set_postfix({"tr_loss": np.mean(train_losses)})
        
        scheduler.step()
        with torch.no_grad():
            val_losses = []
            model.eval()
            for batch_idx, datum in enumerate(valloader):
                geno, loc = datum["genotype"].to(DEVICE), datum["location"].to(DEVICE)
                val_output = model(geno)
                # val_loss = model.euclidean_distance_loss(data.unscale(val_output), data.unscale(loc))
                val_loss = model.haversine_loss(data.unscale(val_output), data.unscale(loc))
                val_losses.append(val_loss.cpu().numpy())
            pbar.set_postfix({"val_loss": np.mean(val_losses)})
            if np.mean(val_losses) < best_val_loss:
                best_val_loss = np.mean(val_losses)
                torch.save(model.state_dict(), f"/home/cynthia/uw/cefs/{form}_ckpt_{batch_size}.pt")
    
        # record = f"Epoch {epoch}: training - {loss}, val - {val_loss}"

        if logger == "wandb":
            wandb.log({
                "train_loss": np.mean(train_losses), 
                "val_loss": np.mean(val_losses)
                })
    return model


def test(testloader, model, data):
    model.to(DEVICE)
    test_losses = []
    with torch.no_grad():
        for batch_idx, datum in enumerate(testloader):
            geno, loc = datum['genotype'].to(DEVICE), datum['location'].to(DEVICE)
            output = model(geno)
            # loss = model.euclidean_loss(data.unscale(output), data.unscale(loc))
            loss = model.haversine_distance(data.unscale(output), data.unscale(loc))
            test_losses.extend(loss.cpu().numpy())

    print(f"Test loss: {np.mean(test_losses)} over {len(test_losses)} samples")
    np.save(f"/home/cynthia/uw/cefs/{batch_size}_{lr}_{time.time()}.npy", test_losses)



if __name__ == "__main__":
    torch.manual_seed(sys.argv[-1])
    data = AlleleData("~/uw/cefs/REFELE_5.8_filtered_", "/home/cynthia/uw/cefs/zones_44_", seed=sys.argv[-1])
    train_data, test_data, val_data = torch.utils.data.random_split(data, [0.7, 0.1, 0.2])
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True) 
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = GenoLoc(data.cat_summaries)
    # model = torch.jit.script(model)
    # torch.jit.save(model, "/home/cynthia/uw/cefs/genoguesser.pt")
    # raise Exception("done")

    
    model = train(trainloader, valloader, model)
    # model.load_state_dict(torch.load(f"/home/cynthia/uw/cefs/{form}_ckpt_{batch_size}.pt", weights_only=False))
    test(testloader, model, data)

    

    # early_stopping = EarlyStopping('val_loss')

    # trainer = Trainer(max_epochs=20, callbacks=[early_stopping], log_every_n_steps=10)
    # trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader) 
    
