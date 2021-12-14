# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import argparse
from filelock import FileLock
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import ray
from ray import tune
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
import timeit

def init_hook():
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

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


def train(model, optimizer, train_loader, device=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx * len(data) > EPOCH_SIZE:
        #     return
        data, target = data, target
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader, device=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # if batch_idx * len(data) > TEST_SIZE:
            #     break
            data, target = data, target
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def get_data_loaders():
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data",
                train=True,
                download=True,
                transform=mnist_transforms),
            batch_size=64,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data",
                train=False,
                download=True,
                transform=mnist_transforms),
            batch_size=64,
            shuffle=True)
    return train_loader, test_loader



class Training(TrainingOperator):
    def setup(self, config):
        self.train_loader, self.test_loader = get_data_loaders()
        
        model = MLPNet()
        optimizer = optim.SGD(
            model.parameters(), lr=0.01 * config["lr_scaler"])


        self.model, self.optimizer = self.register(
            models=model,
            optimizers=optimizer,
        )
        self.register_data(train_loader=self.train_loader, validation_loader=None)

    def train_epoch(self, *pargs, **kwargs):
        def benchmark():
            train(self.model, self.optimizer, self.train_loader)

        print("Running benchmark...")
        time = timeit.timeit(benchmark, number=args.num_batches_per_iter)
        img_sec = args.batch_size * args.num_batches_per_iter / time

        acc = test(self.model, self.test_loader)
        return {"img_sec": img_sec, "acc": acc}





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=True,
        help="Enables GPU training")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--ray-address",
        help="Address of Ray cluster for seamless distributed execution.")
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using "
        "Ray Client.")

    parser.add_argument(
        "--batch-size", type=int, default=32, help="input batch size")

    parser.add_argument(
        "--num-batches-per-iter",
        type=int,
        default=10,
        help="number of batches per benchmark iteration")

    parser.add_argument(
        "--num-iters", type=int, default=10, help="number of benchmark iterations")

    parser.add_argument(
        "--local",
        action="store_true",
        default=True,
        help="Disables cluster training")

    args, _ = parser.parse_known_args()

    if args.server_address:
        ray.init(f"ray://{args.server_address}")
    elif args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init(num_cpus=2 if args.smoke_test else None)

    args.cuda = args.cuda and torch.cuda.is_available()
    device = "GPU" if args.cuda else "CPU"

    num_workers = 1 if args.local else int(ray.cluster_resources().get(device))

    print("Batch size: %d" % args.batch_size)
    print("Number of %ss: %d" % (device, num_workers))

    trainer = TorchTrainer(
        training_operator_cls=Training,
        initialization_hook=init_hook,
        config={
            "lr_scaler": num_workers,
        },
        num_workers=num_workers,
        use_gpu=args.cuda,
    )


    img_secs = []
    for x in range(args.num_iters):
        result = trainer.train()
        # print(result)
        img_sec = result["img_sec"]
        print("Iter #%d: %.1f img/sec per %s" % (x, img_sec, device))
        img_secs.append(img_sec)

    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    print(f"Img/sec per {device}: {img_sec_mean:.1f} +-{img_sec_conf:.1f}")
    print("Total img/sec on %d %s(s): %.1f +-%.1f" %
          (num_workers, device, num_workers * img_sec_mean,
           num_workers * img_sec_conf))