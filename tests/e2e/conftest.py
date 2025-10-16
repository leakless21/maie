import os
import time
from pathlib import Path

import pytest
import requests


@pytest.fixture(scope="session")
def api_base_url():
    return os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def api_key():
    return os.getenv("SECRET_API_KEY", "test-key")


@pytest.fixture(scope="session")
def test_assets_dir():
    return Path(__file__).parent / "assets"


@pytest.fixture
def api_client(api_base_url, api_key):
    """API client fixture for E2E tests"""

    class APIClient:
        def __init__(self, base_url, key):
            self.base_url = base_url
            self.session = requests.Session()
            self.session.headers.update({"X-API-Key": key})

        def submit_audio(
            self, file_path, features=None, template_id=None, asr_backend="whisper"
        ):
            with open(file_path, "rb") as f:
                files = {"file": f}
                data = {}
                if features:
                    data["features"] = str(features).replace("'", '"')
                if template_id:
                    data["template_id"] = template_id
                if asr_backend:
                    data["asr_backend"] = asr_backend

                response = self.session.post(
                    f"{self.base_url}/v1/process", files=files, data=data
                )
                return response

        def get_status(self, task_id):
            return self.session.get(f"{self.base_url}/v1/status/{task_id}")

        def wait_for_completion(self, task_id, timeout=300, poll_interval=5):
            start_time = time.time()
            while time.time() - start_time < timeout:
                response = self.get_status(task_id)
                if response.status_code == 200:
                    data = response.json()
                    if data["status"] in ["COMPLETE", "FAILED"]:
                        return data
                time.sleep(poll_interval)
            raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

    return APIClient(api_base_url, api_key)
