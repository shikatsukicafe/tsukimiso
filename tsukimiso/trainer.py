import io
import websocket
import torch
import uuid
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from logging import getLogger
from abc import ABCMeta, abstractmethod
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tsukimiso.context import TkmArguments
from tsukimiso.pb import TkmParams


class TkmTrainer(metaclass=ABCMeta):
    def __init__(self, args: TkmArguments, model: nn.Module, optimizer: Optimizer, loss_func: nn.Module,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader):
        self._logger = getLogger("TkmTrainer")
        self._client_id = str(uuid.uuid4())
        self._device = args.device
        self._round = 0
        self._comm_round = args.round
        self._repo_name = args.repo_name
        self._server_url = args.server_url
        self._loss_func = loss_func
        self._batch_size = args.batch_size
        self._dp_on = args.dp

        self._model = model
        self._optimizer = optimizer
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader

        if self._dp_on:
            self._noise_multiplier = get_noise_multiplier(
                target_epsilon=args.dp_target_eps,
                target_delta=args.dp_target_delta if args.dp_target_delta is not None else 1 / len(train_dataloader.dataset),
                sample_rate=args.batch_size / len(train_dataloader.dataset),
                epochs=self._comm_round,
                accountant="rdp"
            )
            print(f"Calculated Noise Multiplier (σ): {self._noise_multiplier:.4f}")
            privacy_engine = PrivacyEngine()
            self._model, self._optimizer, self._train_dataloader = privacy_engine.make_private(
                module=self._model,
                optimizer=self._optimizer,
                data_loader=self._train_dataloader,
                noise_multiplier=self._noise_multiplier,
                max_grad_norm=1.0,
            )

        ws_target_url = "ws://%s/plugin/ws" % self.server_url
        self._ws = websocket.create_connection(ws_target_url)
        self._model.load_state_dict(self.hello())

    def run(self):
        for _ in range(self._comm_round):
            if self._dp_on:
                with BatchMemoryManager(
                        data_loader=self.train_dataloader,
                        max_physical_batch_size=self._batch_size,
                        optimizer=self.optimizer
                ) as memory_safe_data_loader:
                    self._train_dataloader = memory_safe_data_loader
                    self.model.train().to(self.device)
                    train_metrics = self.train()
            else:
                self.model.train().to(self.device)
                train_metrics = self.train()

            self.model.eval()
            valid_metrics = self.valid()

            self.model.train().to("cpu")
            state_dict = self.update(self.model.state_dict())
            self.model.load_state_dict(state_dict)

            print("[Rnd=%d Cl:%s] tra: %s  / val: %s" % (self._round, self.client_id, str(train_metrics), str(valid_metrics)))

    @abstractmethod
    def train(self) -> dict:
        pass

    @abstractmethod
    def valid(self) -> dict:
        pass

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def loss_func(self) -> nn.Module:
        return self._loss_func

    @property
    def repo_name(self) -> str:
        return self._repo_name

    @property
    def server_url(self) -> str:
        return self._server_url

    @property
    def client_id(self) -> str:
        return self._client_id

    @property
    def device(self):
        return self._device

    @property
    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    @property
    def valid_dataloader(self) -> DataLoader:
        return self._valid_dataloader

    @property
    def round(self) -> int:
        return self._round

    @torch.no_grad()
    def hello(self) -> dict:
        # send hello
        msg = TkmParams()
        msg.op = "hello"
        self._ws.send(msg.SerializeToString(), websocket.ABNF.OPCODE_BINARY)

        # get initialized model
        message = self._ws.recv()
        msg.ParseFromString(message)
        buf = io.BytesIO(msg.params)
        return torch.load(buf)

    @torch.no_grad()
    def update(self, model_state_dict: dict) -> dict:
        buf = io.BytesIO()
        torch.save(model_state_dict, buf)

        # send update
        msg = TkmParams()
        msg.op = "update"
        msg.id = self._client_id
        msg.params = buf.getvalue()
        self._ws.send(msg.SerializeToString(), websocket.ABNF.OPCODE_BINARY)

        # get aggregated model
        message = self._ws.recv()
        msg.ParseFromString(message)
        buf = io.BytesIO(msg.params)

        self._round += 1
        return torch.load(buf)
