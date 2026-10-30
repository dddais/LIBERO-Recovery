"""Optional websocket client without heartbeat deadlines during blocking rendering."""
import time
import websockets.sync.client
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


class NoPingClientPolicy(WebsocketClientPolicy):
    def _wait_for_server(self):
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers,
                    ping_interval=None, ping_timeout=None,
                )
                return conn, msgpack_numpy.unpackb(conn.recv())
            except ConnectionRefusedError:
                time.sleep(5)
