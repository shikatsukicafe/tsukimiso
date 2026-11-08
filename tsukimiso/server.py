import queue
import contextlib
import time
import threading
import uvicorn
from typing import Optional
from logging import getLogger
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool
from .pb import TkmParams, TkmMetrics


class RepoInfo(BaseModel):
    round: int = 0
    metrics: Optional[dict] = None


class RepoManager:
    def __init__(self):
        self._logger = getLogger("Tsukimiso-Server")
        self._queue = queue.Queue()
        self._round = 0
        self._pushed_cnt = 0
        self._clients = {}
        self._metrics = {}
        self._logger.info("initialize complete!!")

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self._clients[client_id] = {"ws": websocket, "round": 0, "metrics": {}, "waiting": False}

    def disconnect(self, client_id: str):
        self._clients.pop(client_id)

    def hello(self):
        ret = TkmParams()
        ret.op = "hello"
        ret.params = bytes()
        if self._queue.qsize() > 0:
            ret.params = self._queue.get()
        return ret

    def pull(self) -> TkmParams:
        ret = TkmParams()
        ret.op = "pull"
        ret.params = bytes()
        if self._queue.qsize() > 0:
            ret.params = self._queue.get()
        return ret

    def push(self, params: bytes):
        self._queue.put(params)
        self._pushed_cnt += 1

    async def update(self, client_id: str, round_num: int, metrics: dict):
        self._clients[client_id]["round"] = round_num
        self._clients[client_id]["metrics"] = metrics
        self._clients[client_id]["waiting"] = True

        all_received = True
        for c in self._clients.values():
            if not c["waiting"]:
                all_received = False
                break

        if all_received:
            metrics = {}
            client_num = len(self._clients)
            for i, c in enumerate(self._clients.values()):
                for k, v in c["metrics"].items():
                    if i == 0:
                        metrics[k] = 0.
                    metrics[k] += v / client_num

            self._round += 1
            self._metrics = metrics
            self._logger.info("[ROUND-%03d] PUSHED= %8d\t metrics: %s" % (self._round, self._pushed_cnt, metrics))

            ret = TkmParams()
            ret.op = "update"
            for c in self._clients.values():
                await c["ws"].send_bytes(ret.SerializeToString())
                c["waiting"] = False


app = FastAPI()
repo = RepoManager()


@app.websocket("/ws/{client_id}")
async def ws_endpoint(websocket: WebSocket, client_id: str):
    await repo.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_bytes()
            msg = TkmParams()
            await run_in_threadpool(msg.ParseFromString, data)
            if msg.op == "hello":
                # hello
                ret = repo.hello()
            elif msg.op == "pull":
                # pull
                ret = repo.pull()
                bytes_data = await run_in_threadpool(ret.SerializeToString)
                await websocket.send_bytes(bytes_data)

                # push
                data = await websocket.receive_bytes()
                msg = TkmParams()
                await run_in_threadpool(msg.ParseFromString, data)
                repo.push(msg.params)

            elif msg.op == "update":
                metrics = {metrics.name: metrics.value for metrics in msg.stats.metrics}
                await repo.update(client_id, msg.stats.round, metrics)

            else:
                # sequential error
                pass

    except WebSocketDisconnect:
        repo.disconnect(client_id)


class ConLServer(uvicorn.Server):
    def __init__(self, host: str, port_no: int, ws_max_size: int = (1000 * 1024 * 1024),
                 ws_ping_interval: float = 120.0, ws_ping_timeout: float = 120.0, **kwargs):
        config = uvicorn.Config(app, host=host, port=port_no, ws_max_size=ws_max_size,
                                ws_ping_interval=ws_ping_interval, ws_ping_timeout=ws_ping_timeout, access_log=False, **kwargs)
        super().__init__(config)

    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started and thread.is_alive():
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()
