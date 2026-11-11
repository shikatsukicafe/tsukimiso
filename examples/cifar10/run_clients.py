import logging
import argparse
import torch.nn as nn
from torchvision import models
from torchvision import datasets
from torchvision.transforms import ToTensor
from logging import basicConfig, getLogger
from multiprocessing import Process, set_start_method, get_start_method
from tsukimiso import FedDatasetsClassification, TkmArguments, FedAvg

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
basicConfig(level=logging.INFO, format=formatter)
logger = getLogger("Frose-Runner")


def _proc_run(conf: TkmArguments, client_id: int, dataset, device="cpu"):
    model = models.resnet18()
    optimizer = FedAvg(conf, model.parameters(), lr=0.1, weight_decay=0.01)
    state_dict = optimizer.hello()
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()
    while optimizer.round <= conf.round:
        logger.info("[Client:%4d]  Round-%d Start!!" % (client_id, optimizer.round))
        model.train().to(device)
        batch_loss = []
        for batch_idx, (x, labels) in enumerate(dataset["data"]):
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            labels = labels.long()

            log_probs = model(x)
            loss = criterion(log_probs, labels)  # pylint: disable=E1102
            loss.backward()
            batch_loss.append(loss.item())

            optimizer.step()

        if len(batch_loss) > 0:
            logger.info("[Client:%4d]    Loss: %.8f" % (client_id, sum(batch_loss) / len(batch_loss)))

        # round update
        state_dict = optimizer.update(model.state_dict())
        model.load_state_dict(state_dict)

    # close
    logger.info("[Client:%4d]  Training Finished!!" % (client_id,))


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_path", type=str, help="path of config file")
    args = arg_parser.parse_args()
    conf = TkmArguments.from_yml(args.config_path)

    train_data = datasets.CIFAR10(root=conf.data_cache_dir, train=True, download=True, transform=ToTensor())
    valid_data = datasets.CIFAR10(root=conf.data_cache_dir, train=False, download=True, transform=ToTensor())

    fed_datasets = FedDatasetsClassification(conf.worker_num, conf.batch_size, conf.inner_loop,
                                             conf.partition_method, conf.partition_alpha,
                                             train_data, valid_data, 10)

    if get_start_method() == 'fork':
        set_start_method('spawn', force=True)

    clients = []
    for client_id in range(conf.worker_num):
        client = Process(target=_proc_run,
                         args=(conf, client_id, fed_datasets.fed_dataset(client_id), conf.device,))

        # client start
        client.start()
        clients.append(client)

    # wait for stop clients
    for client in clients:
        client.join()


if __name__ == "__main__":
    main()
