# Author: Gentry Atkinson
# Organization: St. Edwards University
# Date: 15 Mar, 2024

import os
from transformers import pipeline
import numpy as np
import scipy

pipe = pipeline(
    "text-to-audio",
    model="facebook/musicgen-small",
)

output = pipe("A jazzy 10-second intro song for a science focused podcast")

audio = output["audio"]
#Channels Last
audio = audio.squeeze()
audio = np.moveaxis(audio, -1, 0)
scipy.io.wavfile.write(
        os.path.join("Podcast Generator", "tunes", f"themesong.wav"), 
        rate=output['sampling_rate'], data=audio.astype(np.float32)
)
