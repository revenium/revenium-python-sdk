"""The LiteLLM proxy's app, with the SDK's metering drain in its shutdown.

Serve this app in place of ``litellm.proxy.proxy_server:app`` so a restart
sends the usage records the SDK still holds::

    CONFIG_FILE_PATH=config.yaml uvicorn revenium_middleware.litellm.proxy.asgi:app --host 0.0.0.0 --port 4000

See the README's "Restarts and shutdown" under LiteLLM Proxy Mode.
"""

from litellm.proxy.proxy_server import app as litellm_proxy_app

from .shutdown import DrainMeteringOnShutdown

app = DrainMeteringOnShutdown(litellm_proxy_app)

__all__ = ["app"]
