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
import random

# Setup
USE_CPU = False

if torch.cuda.is_available() and not USE_CPU:
    device = "cuda:1"
else:
    device = "cpu"

# Setup

input_text = ""
filename = f"episode_{datetime.now()}"
title_prompt = "Generate a title for one episode of a podcast that is name 'No Humans Made this Podcast'. This podcast is about AI and society. The title should be fun and witty."
script = ""

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
    {"role": "user", "content": f"{title_prompt}"},
]
outputs = pipe(
    messages,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    stop_sequence="<|im_end|>",
)

#Get a title for the episode

title = outputs[0]["generated_text"][-1]["content"]

#Generate Script text from several prompts

all_segments = ["Tech_Tales", "AI_Q_and_A", "AI_Fails", "Future_Forcast", 
                "AI_and_Pop_Culture", "Tech_Trivia", "Listener_Stories"
                ]

segments = ["Introduction"]
segments.extend(random.sample(all_segments, 4))
segments.extend(["Today's Sponsor", "Conclusion"])

for segment in segments:

    with open(os.path.join("Podcast Generator",f"{segment}_prompt.txt")) as f:
        input_text = f.read()
    input_text.replace("{title}", title)
    messages = [
    {
        "role": "system",
        "content": "You are a funny and exciting podcast host.",
    },
    {"role": "user", "content": f"{title_prompt}"},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=5096,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        stop_sequence="<|im_end|>",
    )
    scrip_segment = outputs[0]["generated_text"][-1]["content"]

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
audio = speech["audio"]
audio = np.moveaxis(audio, -1, 0)
#audio = np.interp(audio, (audio.min(), audio.max()), (0, 65535))
np.save(os.path.join("Podcast Generator", "episode_audio", f"{filename}.npy"), audio)
scipy.io.wavfile.write(os.path.join("Podcast Generator", "episode_audio", f"{filename}.wav"), rate=speech["sampling_rate"], data=audio.astype(np.float32))
print("Audio Saved")

