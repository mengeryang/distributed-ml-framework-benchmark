import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.nn import functional as F
from ray_lightning import RayPlugin
import ray

class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super(LightningMNISTClassifier, self).__init__()

        self.data_dir = data_dir
        self.lr = config["lr"]
        layer_1, layer_2 = config["layer_1"], config["layer_2"]
        self.batch_size = config["batch_size"]

        

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, layer_1)
        self.layer_2 = torch.nn.Linear(layer_1, layer_2)
        self.layer_3 = torch.nn.Linear(layer_2, 10)
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)
        x = F.softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        acc = self.accuracy(logits, y)
        return {"val_loss": loss, "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def prepare_data(self):
        self.dataset = MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor())

    def train_dataloader(self):
        dataset = self.dataset
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset, [train_length - 5000, 5000],
            generator=torch.Generator().manual_seed(0))
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            num_workers=1,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        dataset = self.dataset
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset, [train_length - 5000, 5000],
            generator=torch.Generator().manual_seed(0))
        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            num_workers=1,
            drop_last=True,
            pin_memory=True,
        )




config={
    "lr": 1e-4,
    "layer_1": 256,
    "layer_2": 128,
    "batch_size": 256
}


# # Instantiate model
# model = LightningMNISTClassifier(config, data_dir="./")

# # Create Trainer and start training
# trainer = pl.Trainer( max_epochs=10)
# trainer.fit(model)

# variables for Ray around parallelism and hardware
num_workers = 8
use_gpu = False

# Initialize ray.
ray.init()

model = LightningMNISTClassifier(config, data_dir="./")

trainer = pl.Trainer(
    max_epochs=10, 
    plugins=[RayPlugin(num_workers=num_workers, use_gpu=use_gpu)])
trainer.fit(model)