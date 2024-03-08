# Author: Gentry Atkinson
# Organization: St. Edwards University
# Date: 07 Mar, 2024


from transformers import pipeline
import torch
import os
from datetime import datetime
import gc
from datasets import load_dataset
import numpy as np

# Setup
USE_CPU = False

if torch.cuda.is_available() and not USE_CPU:
    device = "cuda:1"
else:
    device = "cpu"

# Generate a script from a prompt

input_text = "Write a new birthday song."
with open(os.path.join("Podcast Generator","prompt.txt")) as f:
    input_text = f.read()

pipe = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-gemma-v0.1",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
messages = [
    {
        "role": "system",
        "content": "You are a funny and exciting podcast host.",
    },
    {"role": "user", "content": f"{input_text}"},
]
outputs = pipe(
    messages,
    max_new_tokens=102400,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    stop_sequence="<|im_end|>",
)
script = outputs[0]["generated_text"][-1]["content"]
filename = f"episode_{datetime.now()}"

with open(os.path.join("Podcast Generator", "scripts", f"{filename}.txt"), 'w+') as f:
    f.write(script)

del pipe
del outputs
del messages
gc.collect()

print("Script Generated")

# Convert Script to Audio
from transformers import pipeline
import scipy

synthesiser = pipeline(
    "text-to-speech", "suno/bark",
)

speech = synthesiser(script, forward_params={"do_sample": True})
print("Audio Generated")
audio = speech["audio"].flatten()
audio = np.interp(audio, (audio.min(), audio.max()), (0, 65535))
scipy.io.wavfile.write(os.path.join("Podcast Generator", "episode_audio", f"{filename}.wav"), rate=speech["sampling_rate"], data=audio)
print("Audio Saved")

