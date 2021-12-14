from argparse import ArgumentParser
from pprint import pprint
import pytorch_lightning as pl
from model import LitMNIST
from data import MNISTDataModule
from ray_lightning import RayPlugin
import ray
import time

def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--ray_accelerator_num_workers", type=int, default=1)
    parser.add_argument("--ray_accelerator_cpus_per_worker", type=int, default=2)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_epoch", type=int, default=3)
    
    parser = LitMNIST.add_model_specific_args(parser)
    parser = MNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    #ray.init(address='auto')
    ray.init()
    # ray.init(address='auto', _redis_password='5241590000000000')

    print('''This cluster consists of
        {} nodes in total
        {} CPU resources in total
        {} GPU resources in total
    '''.format(len(ray.nodes()), ray.cluster_resources()['CPU'], ray.cluster_resources()['GPU']))

    # ------------
    # data
    # ------------
    dm = MNISTDataModule(
        data_dir='', val_split=5000, num_workers=1,
        normalize=False, seed=42, batch_size=64
    )

    # ------------
    # model
    # ------------
    model = LitMNIST(args.hidden_dim, args.learning_rate)

    # ------------
    # training
    # ------------
    # ray.init()


    plugin = RayPlugin(num_workers=args.ray_accelerator_num_workers,
                       num_cpus_per_worker=args.ray_accelerator_cpus_per_worker,
                       use_gpu=args.use_gpu
                       )
    trainer = pl.Trainer(
        gpus=args.gpus, precision=args.precision, plugins=[plugin],
        max_epochs=args.max_epoch
    )

    start_time = time.time()
    trainer.fit(model, datamodule=dm)
    print("--- %s seconds ---" % (time.time() - start_time))

    # ------------
    # testing
    # ------------
    result = trainer.test(model, datamodule=dm)
    pprint(result)


if __name__ == '__main__':
    cli_main()
