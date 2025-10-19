"""Quick debug script to see what's failing."""
from io import BytesIO
from litestar import Litestar
from litestar.testing import TestClient
from src.api.routes import ProcessController

# Create minimal app with debug mode
app = Litestar(route_handlers=[ProcessController], debug=True)

# Create valid audio file
wav_header = (
    b"RIFF"
    + (100).to_bytes(4, "little")
    + b"WAVE"
    + b"fmt "
    + (16).to_bytes(4, "little")
    + (1).to_bytes(2, "little")
    + (1).to_bytes(2, "little")
    + (16000).to_bytes(4, "little")
    + (32000).to_bytes(4, "little")
    + (2).to_bytes(2, "little")
    + (16).to_bytes(2, "little")
    + b"data"
    + (64).to_bytes(4, "little")
    + b"\x00" * 64
)
valid_audio_file = BytesIO(wav_header)

with TestClient(app=app) as client:
    response = client.post(
        "/v1/process",
        files={"file": ("test.wav", valid_audio_file, "audio/wav")},
        data={
            "features": ["clean_transcript", "summary"],
            "template_id": "meeting_notes_v1",
        },
        headers={"X-API-Key": "dev_api_key_change_in_production"},
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

