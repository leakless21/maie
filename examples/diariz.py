import torch
from pyannote.audio import Pipeline

# Use pyannote/speaker-diarization-3.1 from HuggingFace
# It will use cached models from ~/.cache/huggingface/hub if available
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipeline.to(torch.device("cuda"))

# run the pipeline locally on your computer
output = pipeline("tests/assets/Northern Female 1.wav")

# print the predicted speaker diarization
# In pyannote 3.x, output is directly an Annotation object
for turn, _, speaker in output.itertracks(yield_label=True):
    print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")
