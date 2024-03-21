# Author: Gentry Atkinson
# Organization: St. Edwards University
# Date: 15 Mar, 2024

import os
from transformers import pipeline
import numpy as np
import scipy

def generate_theme_song():
    pipe = pipeline(
        "text-to-audio",
        model="facebook/musicgen-small",
    )

    output = pipe("An upbeat, jazzy intro song for a science focused podcast", forward_params={"do_sample": True})

    audio = output["audio"]
    #Channels Last
    audio = audio.squeeze()
    scipy.io.wavfile.write(
            os.path.join("Podcast Generator", "tunes", "themesong.wav"), 
            rate=output['sampling_rate'], data=audio.astype(np.float32)
    )
    # Shorten song to 6 seconds (assuming 30 second original length)
    audio = audio[0:len(audio)//5]
    # Linear fade out last two seconds
    audio_length = len(audio)
    fade_length = audio_length//3
    fade_start = 2*audio_length//3
    for i in range(fade_start, audio_length):
        audio[i] *= (1 - ((i - fade_start)/fade_length))
    np.save(os.path.join("Podcast Generator", "tunes", "shortened_themesong.npy"), audio)
    scipy.io.wavfile.write(
            os.path.join("Podcast Generator", "tunes", "shortened_themesong.wav"), 
            rate=output['sampling_rate'], data=audio.astype(np.float32)
    )

def generate_transistion_song():
    pipe = pipeline(
        "text-to-audio",
        model="facebook/musicgen-small",
    )

    output = pipe("A mellow but catchy transition tune to separate the segments of a podcate", forward_params={"do_sample": True})

    audio = output["audio"]
    #Channels Last
    audio = audio.squeeze()
    scipy.io.wavfile.write(
            os.path.join("Podcast Generator", "tunes", "transition.wav"), 
            rate=output['sampling_rate'], data=audio.astype(np.float32)
    )
    # Shorten song to 3 seconds (assuming 30 second original length)
    audio = audio[0:len(audio)//10]
    # Linear fade out last two seconds
    audio_length = len(audio)
    fade_length = audio_length//3
    fade_start = 2*audio_length//3
    for i in range(fade_start, audio_length):
        audio[i] *= (1 - ((i - fade_start)/fade_length))
    np.save(os.path.join("Podcast Generator", "tunes", "shortened_transition.npy"), audio)
    scipy.io.wavfile.write(
            os.path.join("Podcast Generator", "tunes", "shortened_transition.wav"), 
            rate=output['sampling_rate'], data=audio.astype(np.float32)
    )

def generate_outro_song():
    pipe = pipeline(
        "text-to-audio",
        model="facebook/musicgen-small",
    )

    output = pipe("A mellow outro song with a strong bassline for a science focused podcast", forward_params={"do_sample": True})

    audio = output["audio"]
    #Channels Last
    audio = audio.squeeze()
    scipy.io.wavfile.write(
            os.path.join("Podcast Generator", "tunes", "outro.wav"), 
            rate=output['sampling_rate'], data=audio.astype(np.float32)
    )
    # Shorten song to 10 seconds (assuming 30 second original length)
    audio = audio[0:len(audio)//3]
    # Linear fade out last two seconds
    audio_length = len(audio)
    fade_length = audio_length//3
    fade_start = 2*audio_length//3
    for i in range(fade_start, audio_length):
        audio[i] *= (1 - ((i - fade_start)/fade_length))
    np.save(os.path.join("Podcast Generator", "tunes", "shortened_outro.npy"), audio)
    scipy.io.wavfile.write(
            os.path.join("Podcast Generator", "tunes", "shortened_outro.wav"), 
            rate=output['sampling_rate'], data=audio.astype(np.float32)
    )

if __name__ == "__main__":
    # generate_theme_song()
    # generate_transistion_song()
    generate_outro_song()
