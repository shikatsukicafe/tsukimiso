import time
import io
import pickle
import websocket
import torch
import uuid
from typing import Dict
from torch import nn
from torch.optim.optimizer import Optimizer
from logging import getLogger
from abc import ABCMeta, abstractmethod
from ..context import TkmArguments
from ..pb import TkmParams


class TsukimiOptFrame(Optimizer, metaclass=ABCMeta):
    def __init__(self, params, defaults, args: TkmArguments):
        super().__init__(params, defaults)
        self._client_id = str(uuid.uuid4())
        self._round = 0
        self._repo_name = args.repo_name
        self._server_url = args.server_url

        ws_target_url = "ws://%s/ws/%s" % (self.server_url, self.client_id)
        self._ws = websocket.create_connection(ws_target_url)
        self._logger = getLogger("TsukimiOptimizer")

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
        buf = io.BytesIO(pickle.loads(msg.params))
        return torch.load(buf)

    @torch.no_grad()
    def update(self, model_state_dict: dict) -> dict:
        buf = io.BytesIO()
        torch.save(model_state_dict, buf)

        # send update
        msg = TkmParams()
        msg.op = "update"
        msg.params = buf.getvalue()
        self._ws.send(msg.SerializeToString(), websocket.ABNF.OPCODE_BINARY)

        # get aggregated model
        message = self._ws.recv()
        msg.ParseFromString(message)
        buf = io.BytesIO(pickle.loads(msg.params))
        return torch.load(buf)

    @abstractmethod
    def snd_params(self) -> Dict:
        return {}

    @abstractmethod
    def rcv_params(self, others: Dict):
        pass

