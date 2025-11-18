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
    return os.getenv("SECRET_API_KEY", "test-key-123456789012345678901234567890")


@pytest.fixture(scope="session")
def test_assets_dir():
    return Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def verify_api_running(api_base_url):
    """Verify API server is running before running tests"""
    try:
        response = requests.get(f"{api_base_url}/health", timeout=5)
        if response.status_code != 200:
            pytest.exit(
                f"API server at {api_base_url} returned status {response.status_code}. "
                f"Start the API with: ./scripts/dev.sh --api-only",
                returncode=1,
            )
    except requests.exceptions.ConnectionError:
        pytest.exit(
            f"API server not running at {api_base_url}. "
            f"Start it with: ./scripts/dev.sh --api-only",
            returncode=1,
        )
    except Exception as e:
        pytest.exit(f"Failed to connect to API: {e}", returncode=1)


@pytest.fixture(scope="session")
def verify_workers_running():
    """Verify that RQ workers are running before E2E tests"""
    try:
        import redis
        from rq import Queue

        # Connect to the queue Redis (DB 0)
        r = redis.Redis(
            host=os.getenv("REDIS_QUEUE_HOST", "localhost"),
            port=int(os.getenv("REDIS_QUEUE_PORT", 6379)),
            db=0,
        )

        # Check if we can connect
        r.ping()

        # Check for workers
        queue = Queue(name=os.getenv("RQ_QUEUE_NAME", "maie_default"), connection=r)
        workers = queue.connection.smembers("rq:workers")

        if not workers:
            pytest.skip(
                "No RQ workers detected. "
                "E2E tests require workers to process tasks. "
                "Start workers with: ./scripts/dev.sh --worker-only",
                allow_module_level=True,
            )

        print(f"\\n✓ Found {len(workers)} RQ worker(s) running")

    except ImportError:
        # If redis/rq not available in test env, skip check with warning
        print(
            "\\n⚠ Warning: Cannot verify workers (redis/rq not available in test environment)"
        )
    except redis.ConnectionError:
        pytest.exit(
            "\\n\\nCannot connect to Redis queue!\\n"
            "E2E tests require Redis to be running.\\n"
            "Start Redis with: docker-compose up -d redis\\n",
            returncode=1,
        )
    except Exception as e:
        print(f"\\n⚠ Warning: Could not verify workers: {e}")


@pytest.fixture
def api_client(api_base_url, api_key, verify_api_running, verify_workers_running):
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
            last_status = None
            poll_count = 0
            while time.time() - start_time < timeout:
                response = self.get_status(task_id)
                if response.status_code == 200:
                    data = response.json()
                    current_status = data.get("status")

                    # Log status changes
                    if current_status != last_status:
                        elapsed = time.time() - start_time
                        print(
                            f"\n[{elapsed:.1f}s] Task {task_id} status: {current_status}"
                        )
                        last_status = current_status

                    if current_status == "COMPLETE":
                        elapsed = time.time() - start_time
                        print(
                            f"\n[{elapsed:.1f}s] Task {task_id} completed successfully"
                        )
                        return data
                    elif current_status == "FAILED":
                        # Fail fast with clear error message
                        error_msg = data.get("error", "Unknown error")
                        raise AssertionError(
                            f"Task {task_id} failed: {error_msg}\nFull response: {data}"
                        )

                    # Show progress every 10 polls
                    poll_count += 1
                    if poll_count % 10 == 0:
                        elapsed = time.time() - start_time
                        print(
                            f"\n[{elapsed:.1f}s] Still waiting for task {task_id}... (status: {current_status})"
                        )
                elif response.status_code == 404:
                    raise AssertionError(
                        f"Task {task_id} not found in Redis. Check if task was created properly."
                    )
                else:
                    raise AssertionError(
                        f"Unexpected status code {response.status_code} for task {task_id}"
                    )

                time.sleep(poll_interval)

            elapsed = time.time() - start_time
            raise TimeoutError(
                f"Task {task_id} did not complete within {timeout}s\n"
                f"Last status: {last_status}\n"
                f"This usually means workers are not running or not processing tasks.\n"
                f"Start workers with: ./scripts/dev.sh --worker-only"
            )

    return APIClient(api_base_url, api_key)
