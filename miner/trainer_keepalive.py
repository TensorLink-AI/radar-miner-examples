"""Periodic keepalive pings for Basilica trainer pods.

Prevents idle-scaling (503s) by pinging each active trainer endpoint
on a regular interval. Designed to run as a background thread alongside
the miner neuron's main loop.

Usage:
    from miner.trainer_keepalive import KeepaliveManager

    manager = KeepaliveManager(interval=60)
    manager.start()

    # When a trainer pod is deployed:
    manager.register("https://<uuid>.deployments.basilica.ai")

    # When the pod is torn down:
    manager.unregister("https://<uuid>.deployments.basilica.ai")

    # On shutdown:
    manager.stop()
"""

import logging
import threading
import time
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 60  # seconds between pings
DEFAULT_TIMEOUT = 10   # seconds per HTTP request
HEALTH_PATH = "/health"


class KeepaliveManager:
    """Background thread that pings active Basilica trainer endpoints."""

    def __init__(self, interval: int = DEFAULT_INTERVAL, timeout: int = DEFAULT_TIMEOUT):
        self._interval = interval
        self._timeout = timeout
        self._endpoints: dict[str, dict] = {}  # url -> {failures, last_ping, ...}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def interval(self) -> int:
        return self._interval

    @property
    def endpoints(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._endpoints)

    def register(self, url: str) -> None:
        """Add a trainer endpoint to the keepalive rotation."""
        url = url.rstrip("/")
        with self._lock:
            if url not in self._endpoints:
                self._endpoints[url] = {
                    "consecutive_failures": 0,
                    "last_ping": 0.0,
                    "last_status": None,
                    "total_pings": 0,
                }
                logger.info("Registered keepalive endpoint: %s", url)

    def unregister(self, url: str) -> None:
        """Remove a trainer endpoint from the keepalive rotation."""
        url = url.rstrip("/")
        with self._lock:
            removed = self._endpoints.pop(url, None)
        if removed:
            logger.info("Unregistered keepalive endpoint: %s", url)

    def start(self) -> None:
        """Start the background keepalive thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="keepalive")
        self._thread.start()
        logger.info("Keepalive thread started (interval=%ds)", self._interval)

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5)
            self._thread = None
        logger.info("Keepalive thread stopped")

    def ping(self, url: str) -> int | None:
        """Send a single keepalive ping. Returns HTTP status or None on error."""
        ping_url = url.rstrip("/") + HEALTH_PATH
        try:
            req = urllib.request.Request(ping_url, method="GET")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                status = resp.status
            return status
        except urllib.error.HTTPError as exc:
            return exc.code
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            logger.warning("Keepalive ping failed for %s: %s", url, exc)
            return None

    def _run(self) -> None:
        """Main loop: ping all registered endpoints every interval."""
        while not self._stop_event.is_set():
            with self._lock:
                urls = list(self._endpoints.keys())

            for url in urls:
                if self._stop_event.is_set():
                    break
                status = self.ping(url)
                self._record_result(url, status)

            self._stop_event.wait(timeout=self._interval)

    def _record_result(self, url: str, status: int | None) -> None:
        """Update tracking state after a ping attempt."""
        with self._lock:
            info = self._endpoints.get(url)
            if info is None:
                return
            info["last_ping"] = time.time()
            info["last_status"] = status
            info["total_pings"] += 1

            if status is not None and 200 <= status < 500:
                info["consecutive_failures"] = 0
            else:
                info["consecutive_failures"] += 1
                logger.warning(
                    "Keepalive unhealthy: %s status=%s (failures=%d)",
                    url, status, info["consecutive_failures"],
                )
