import logging
import argparse
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from logging import basicConfig, getLogger
from multiprocessing import Process, set_start_method, get_start_method
from tsukimiso import FedDatasetsClassification, TkmArguments, TkmTrainer
from model import ResNet18

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
basicConfig(level=logging.INFO, format=formatter)
logger = getLogger("CIFAR10-Trainer")


class Cifar10Trainer(TkmTrainer):
    def train(self)-> dict:
        logger.info("[Client: %s]  Round-%d Start!!" % (self.client_id, self.round))
        batch_loss = []
        for x, labels in self.train_dataloader:
            x, labels = x.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            labels = labels.long()

            log_probs = self.model(x)
            loss = self.loss_func(log_probs, labels)  # pylint: disable=E1102
            loss.backward()
            batch_loss.append(loss.item())

            self.optimizer.step()

        return {"loss": sum(batch_loss) / len(batch_loss)}

    def valid(self) -> dict:
        class_correct = list(0. for _ in range(10))
        class_total = list(0. for _ in range(10))

        batch_loss = []
        with torch.no_grad():
            total = 0
            correct = 0
            for x, target in self.valid_dataloader:
                x = x.to(self.device)
                target = target.to(self.device)
                pred = self.model(x)
                target = target.long()
                loss = self.loss_func(pred, target)  # pylint: disable=E1102
                batch_loss.append(loss.item())

                _, predicted = torch.max(pred, 1)
                c = (predicted == target).squeeze()
                for i in range(4):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                total += target.size(0)
                correct += (predicted == target).sum().item()

            accuracy = correct / total

        return {"accuracy": accuracy, "loss": sum(batch_loss) / len(batch_loss)}


def _proc_run(conf: TkmArguments, dataset):
    model = ResNet18()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)
    trainer = Cifar10Trainer(
        args=conf,
        model=model,
        optimizer=optimizer,
        loss_func=nn.CrossEntropyLoss(),
        train_dataloader=dataset["train_dataloader"],
        valid_dataloader=dataset["valid_dataloader"]
    )
    trainer.run()


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
        client = Process(target=_proc_run, args=(conf, fed_datasets.fed_dataset(client_id),))
        # client start
        client.start()
        clients.append(client)

    # wait for stop clients
    for client in clients:
        client.join()


if __name__ == "__main__":
    main()
