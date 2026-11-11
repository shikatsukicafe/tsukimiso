import io
import torch
from typing import Optional
from torch import nn
from starlette.endpoints import WebSocketEndpoint
from .pb import TkmParams


class TkmSrvMixin(WebSocketEndpoint):
    encoding = "bytes"
    model: Optional[nn.Module] = None
    _active_conn = {}

    async def on_connect(self, websocket):
        await websocket.accept()
        self._active_conn[websocket] = {
            "id": None,
            "params": None,
            "received": False
        }

    async def on_receive(self, websocket, data: bytes):
        model = self.get_model()
        msg = TkmParams()
        msg.ParseFromString(data)
        self._active_conn[websocket]["id"] = msg.id

        if msg.op == "hello":
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            msg.params = buf.getvalue()
            await websocket.send_bytes(msg.SerializeToString())

        elif msg.op == "update":
            buf = io.BytesIO(msg.params)
            self._active_conn[websocket]["params"] = torch.load(buf)
            self._active_conn[websocket]["received"] = True

            if self._is_all_received():
                self.aggregate()

                buf = io.BytesIO()
                torch.save(model.state_dict(), buf)
                msg.params = buf.getvalue()
                await self.broadcast(msg.SerializeToString())

                self._clear_all_received()

    async def on_disconnect(self, websocket, close_code):
        self._active_conn.pop(websocket, None)

    @classmethod
    async def broadcast(cls, message: bytes):
        for connection in cls._active_conn:
            await connection.send_bytes(message)

    @classmethod
    def get_model(cls) -> nn.Module:
        if cls.model is None:
            raise ValueError("model is not initialized")
        return cls.model

    @classmethod
    def client_num(cls) -> int:
        return len(cls._active_conn)

    @classmethod
    def _is_all_received(cls):
        return all(conn["received"] for conn in cls._active_conn.values())

    @classmethod
    def _clear_all_received(cls):
        for conn in cls._active_conn.values():
            conn["received"] = False

    @classmethod
    def aggregate(cls):
        model = cls.get_model()

        with torch.no_grad():
            average_params = model.cpu().state_dict()
            sample_rate = 1 / cls.client_num()
            for i, v in enumerate(cls._active_conn.values()):
                for k in average_params.keys():
                    if i == 0:
                        average_params[k] = v["params"][k] * sample_rate
                    else:
                        average_params[k] += v["params"][k] * sample_rate

            model.load_state_dict(average_params)
