import uvicorn
import torch
from starlette.endpoints import WebSocketEndpoint
from safetensors.torch import save, load
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from .pb import TkmParams


class TkmSrvAggregator(WebSocketEndpoint):
    encoding = "bytes"
    _active_conn = {}

    async def on_connect(self, websocket):
        await websocket.accept()
        self._active_conn[websocket] = {
            "id": None,
            "params": None,
            "received": False
        }

    async def on_receive(self, websocket, data: bytes):
        msg = TkmParams()
        msg.ParseFromString(data)
        self._active_conn[websocket]["id"] = msg.id

        if msg.op == "hello":
            pass

        elif msg.op == "update":
            self._active_conn[websocket]["params"] = load(msg.params)
            self._active_conn[websocket]["received"] = True

            if self._is_all_received():
                msg.params = self.aggregate()
                await self.broadcast(msg.SerializeToString())

                self._clear_all_received()

    async def on_disconnect(self, websocket, close_code):
        self._active_conn.pop(websocket, None)

    @classmethod
    async def broadcast(cls, message: bytes):
        for connection in cls._active_conn:
            await connection.send_bytes(message)

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
    def aggregate(cls) -> bytes:
        with torch.no_grad():
            average_params = {}
            sample_rate = 1 / cls.client_num()
            for i, conn in enumerate(cls._active_conn.values()):
                p = conn["params"]
                for k, v in p.items():
                    if i == 0:
                        average_params[k] = v * sample_rate
                    else:
                        average_params[k] += v * sample_rate

        return save(average_params)


app = Starlette(routes=[WebSocketRoute("/ws", TkmSrvAggregator)])


def run(host: str = "127.0.0.1", port: int = 9200, ws_max_size: int = 1024 * 1024 * 1024):
    uvicorn.run(app, host=host, port=port, ws_max_size=ws_max_size, ws_ping_interval=120., ws_ping_timeout=120.)
