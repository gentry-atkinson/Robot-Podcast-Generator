# Author: Gentry Atkinson
# Organization: St. Edwards University
# Date: 15 Mar, 2024

import os
from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write

model = MAGNeT.get_pretrained("facebook/audio-magnet-medium")

descriptions = ["A short and jazzy intro song for a podcast"]

wav = model.generate(descriptions)  # generates 2 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(
        os.path.join("Podcast Generator", "tunes", f"themesong.wav"), 
        one_wav.cpu(), model.sample_rate, strategy="loudness"
    )
