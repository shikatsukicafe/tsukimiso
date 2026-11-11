import os
import sys
import uvicorn
import importlib
import importlib.util
from starlette.applications import Starlette
from starlette.routing import Mount, WebSocketRoute
from .aggregator import TkmSrvMixin


def load_extension_route(extension_module: str):
    """拡張モジュールを動的ロードし、ルートを生成して返す関数"""
    if extension_module:
        if extension_module.endswith(".py"):
            file_path = os.path.abspath(extension_module)
            module_name = os.path.splitext(os.path.basename(file_path))[0]

            sys.path.insert(0, os.path.dirname(file_path))

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            else:
                raise ImportError(f"Module not found: {file_path}")

        else:
            importlib.import_module(extension_module)

    subclasses = TkmSrvMixin.__subclasses__()
    endpoint_class = subclasses[-1] if subclasses else TkmSrvMixin

    return [WebSocketRoute("/ws", endpoint=endpoint_class)]


def create_app():
    extension_module = os.getenv("TKM_SRV_MODULE", "server")
    print(f"Loading extension module: {extension_module}")

    ws_routes = load_extension_route(extension_module)
    return Starlette(routes=[Mount("/plugin", routes=ws_routes)])


def run(host: str = "127.0.0.1", port: int = 9200, ws_max_size: int = 1024 * 1024 * 1024):
    app = create_app()
    uvicorn.run(app, host=host, port=port, ws_max_size=ws_max_size)
