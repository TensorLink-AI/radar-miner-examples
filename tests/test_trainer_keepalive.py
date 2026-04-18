"""Tests for miner.trainer_keepalive — KeepaliveManager."""

import sys
import os
import time
import threading
from unittest.mock import patch, MagicMock
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from miner.trainer_keepalive import KeepaliveManager, HEALTH_PATH


class TestRegisterUnregister:
    def test_register_adds_endpoint(self):
        mgr = KeepaliveManager()
        mgr.register("https://abc.deployments.basilica.ai")
        assert "https://abc.deployments.basilica.ai" in mgr.endpoints

    def test_register_strips_trailing_slash(self):
        mgr = KeepaliveManager()
        mgr.register("https://abc.deployments.basilica.ai/")
        assert "https://abc.deployments.basilica.ai" in mgr.endpoints

    def test_register_idempotent(self):
        mgr = KeepaliveManager()
        mgr.register("https://abc.deployments.basilica.ai")
        mgr.register("https://abc.deployments.basilica.ai")
        assert len(mgr.endpoints) == 1

    def test_unregister_removes_endpoint(self):
        mgr = KeepaliveManager()
        mgr.register("https://abc.deployments.basilica.ai")
        mgr.unregister("https://abc.deployments.basilica.ai")
        assert len(mgr.endpoints) == 0

    def test_unregister_nonexistent_is_noop(self):
        mgr = KeepaliveManager()
        mgr.unregister("https://nonexistent.basilica.ai")
        assert len(mgr.endpoints) == 0

    def test_multiple_endpoints(self):
        mgr = KeepaliveManager()
        mgr.register("https://a.basilica.ai")
        mgr.register("https://b.basilica.ai")
        assert len(mgr.endpoints) == 2

    def test_initial_state(self):
        mgr = KeepaliveManager()
        mgr.register("https://abc.basilica.ai")
        info = mgr.endpoints["https://abc.basilica.ai"]
        assert info["consecutive_failures"] == 0
        assert info["last_status"] is None
        assert info["total_pings"] == 0


class TestPing:
    @patch("miner.trainer_keepalive.urllib.request.urlopen")
    def test_ping_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        mgr = KeepaliveManager(timeout=5)
        status = mgr.ping("https://abc.basilica.ai")
        assert status == 200

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.full_url == "https://abc.basilica.ai/health"
        assert req.method == "GET"

    @patch("miner.trainer_keepalive.urllib.request.urlopen")
    def test_ping_http_error_returns_code(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://abc.basilica.ai/health",
            code=503,
            msg="Service Unavailable",
            hdrs={},
            fp=None,
        )
        mgr = KeepaliveManager()
        status = mgr.ping("https://abc.basilica.ai")
        assert status == 503

    @patch("miner.trainer_keepalive.urllib.request.urlopen")
    def test_ping_network_error_returns_none(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        mgr = KeepaliveManager()
        status = mgr.ping("https://abc.basilica.ai")
        assert status is None

    @patch("miner.trainer_keepalive.urllib.request.urlopen")
    def test_ping_timeout_returns_none(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError("timed out")
        mgr = KeepaliveManager()
        status = mgr.ping("https://abc.basilica.ai")
        assert status is None


class TestRecordResult:
    def test_success_resets_failures(self):
        mgr = KeepaliveManager()
        mgr.register("https://abc.basilica.ai")
        # Simulate some failures first
        mgr._record_result("https://abc.basilica.ai", None)
        mgr._record_result("https://abc.basilica.ai", None)
        assert mgr.endpoints["https://abc.basilica.ai"]["consecutive_failures"] == 2

        # Success resets
        mgr._record_result("https://abc.basilica.ai", 200)
        info = mgr.endpoints["https://abc.basilica.ai"]
        assert info["consecutive_failures"] == 0
        assert info["last_status"] == 200
        assert info["total_pings"] == 3

    def test_4xx_resets_failures(self):
        mgr = KeepaliveManager()
        mgr.register("https://abc.basilica.ai")
        mgr._record_result("https://abc.basilica.ai", None)
        # 404 means the server is alive, just no /health endpoint
        mgr._record_result("https://abc.basilica.ai", 404)
        assert mgr.endpoints["https://abc.basilica.ai"]["consecutive_failures"] == 0

    def test_503_increments_failures(self):
        mgr = KeepaliveManager()
        mgr.register("https://abc.basilica.ai")
        mgr._record_result("https://abc.basilica.ai", 503)
        assert mgr.endpoints["https://abc.basilica.ai"]["consecutive_failures"] == 1

    def test_none_increments_failures(self):
        mgr = KeepaliveManager()
        mgr.register("https://abc.basilica.ai")
        mgr._record_result("https://abc.basilica.ai", None)
        assert mgr.endpoints["https://abc.basilica.ai"]["consecutive_failures"] == 1

    def test_unregistered_url_is_noop(self):
        mgr = KeepaliveManager()
        mgr._record_result("https://nonexistent.basilica.ai", 200)  # no error


class TestBackgroundThread:
    @patch("miner.trainer_keepalive.urllib.request.urlopen")
    def test_start_stop(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        mgr = KeepaliveManager(interval=1)
        mgr.register("https://abc.basilica.ai")
        mgr.start()

        assert mgr._thread is not None
        assert mgr._thread.is_alive()

        # Let it run at least one cycle
        time.sleep(0.5)
        mgr.stop()

        assert mgr._thread is None
        assert mgr.endpoints["https://abc.basilica.ai"]["total_pings"] >= 1

    def test_start_idempotent(self):
        mgr = KeepaliveManager(interval=60)
        mgr.start()
        thread1 = mgr._thread
        mgr.start()
        thread2 = mgr._thread
        assert thread1 is thread2
        mgr.stop()

    def test_daemon_thread(self):
        mgr = KeepaliveManager(interval=60)
        mgr.start()
        assert mgr._thread.daemon is True
        mgr.stop()

    @patch("miner.trainer_keepalive.urllib.request.urlopen")
    def test_pings_multiple_endpoints(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        mgr = KeepaliveManager(interval=1)
        mgr.register("https://a.basilica.ai")
        mgr.register("https://b.basilica.ai")
        mgr.start()
        time.sleep(0.5)
        mgr.stop()

        assert mgr.endpoints["https://a.basilica.ai"]["total_pings"] >= 1
        assert mgr.endpoints["https://b.basilica.ai"]["total_pings"] >= 1
