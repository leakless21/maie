# download the pipeline from Huggingface
import torch
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "data/models/pyannote-speaker-diarization-community-1")
pipeline.to(torch.device("cuda"))
# run the pipeline locally on your computer
output = pipeline("tests/assets/weanxinviec.mp3")

# print the predicted speaker diarization 
for turn, speaker in output.speaker_diarization:
    print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")
