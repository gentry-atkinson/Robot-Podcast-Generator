# Author: Gentry Atkinson
# Organization: St. Edwards University
# Date: 07 Mar, 2024


from transformers import pipeline, AutoProcessor, BarkModel
import torch
import os
from datetime import datetime
import gc
import numpy as np
import random
import scipy
import logging

from gen_music import generate_theme_song, generate_transistion_song

logger = logging.getLogger(__name__)
logging.basicConfig(filename='Podcast Generator/logging.txt', level=logging.DEBUG)

def break_text(orig_text: str) -> str:
    """
    Return a block of text with every line <= 256 character
    """
    new_text = ""
    for line in orig_text.split('\n'):
        if len(line) > 256:
            break_points = list(range(256, len(line), 256))
            for break_point in break_points:
                while line[break_point] != ' ':
                    break_point -= 1
                line = line[:break_point] + '\n' + line[break_point+1:]
        new_text += line
        new_text += '\n'
    print(new_text)
    return new_text
            

if __name__ == "__main__":
    # Setup
    USE_CPU = False

    if torch.cuda.is_available() and not USE_CPU:
        device = "cuda"
    else:
        device = "cpu"

    input_text = ""
    filename = f"episode_{datetime.now()}"
    title_prompt = "Generate a title for one episode of a podcast that is named 'No Humans Were Involved with This Podcast'. This podcast is about AI and society. The title should be short and humorous."
    script = {}

    # This isn't necesary. I just think it's neat.
    torch.cuda.init()

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
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        stop_sequence="<|im_end|>",
    )

    #Get a title for the episode

    title = outputs[0]["generated_text"][-1]["content"]
    logger.info(f"Episode title: {title}")

    #Generate Script text from several prompts

    all_segments = ["Tech_Tales", "AI_Q_and_A", "AI_Fails", "Future_Forcast", 
                    "AI_and_Pop_Culture", "Tech_Trivia", "Listener_Stories"
                    ]

    segments = ["Introduction"]
    segments.extend(random.sample(all_segments, 4))
    segments.extend(["Todays_Sponsor", "Conclusion"])

    for segment in segments:

        with open(os.path.join("Podcast Generator", "prompts", f"{segment}_prompt.txt")) as f:
            input_text = f.read()
        input_text = input_text.replace("{title}", title)
        messages = [
        {
            "role": "system",
            "content": "You are a funny and exciting podcast host.",
        },
        {"role": "user", "content": f"{input_text}"},
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
        script_segment = outputs[0]["generated_text"][-1]["content"]
        logger.info(f"{segment} generated. {len(script_segment)} characters")
        script[segment] = break_text(script_segment)

    with open(os.path.join("Podcast Generator", "scripts", f"{filename}.txt"), 'w+') as f:
        f.write('\n'.join(script.values()))
    #print(f"Script finished. Total length: {len(script.split(' '))} words")
    logger.info("Script finished.")


    #Clean up models for memory

    del pipe
    del outputs
    del messages
    gc.collect()

    # Convert Script to Audio

    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")
    all_audio = np.zeros((50, 1))
    # original was speaker 6
    voice_preset = "v2/en_speaker_2"
    if not os.path.isfile(os.path.join("Podcast Generator", "tunes", "shortened_themesong.npy")):
        generate_theme_song()
    theme = np.load(os.path.join("Podcast Generator", "tunes", "shortened_themesong.npy"))
    #Channels last
    if theme.ndim != 2:
        theme = np.reshape(theme, (len(theme), 1))
    all_audio = np.concatenate((all_audio, theme), axis=0)

    if not os.path.isfile(os.path.join("Podcast Generator", "tunes", "shortened_transition.npy")):
        generate_transistion_song()
    transition = np.load(os.path.join("Podcast Generator", "tunes", "shortened_transition.npy"))
    #Channels last
    if theme.ndim != 2:
        theme = np.reshape(theme, (len(theme), 1))

    for title, text in script.items():
        logger.info(f"Reading segment {title}")
        if title != "Introduction":
            all_audio = np.concatenate((all_audio, np.zeros((50, 1))), axis=0)
            all_audio = np.concatenate((all_audio, transition), axis=0)
            inputs = processor(f"Coming up we have {title}", voice_preset=voice_preset)
            audio = model.generate(**inputs)
            audio = audio.cpu().numpy()
            audio = np.moveaxis(audio, -1, 0)
            all_audio = np.concatenate((all_audio, audio), axis=0)
            all_audio = np.concatenate((all_audio, np.zeros((50, 1))), axis=0)
        for i, line in enumerate(text.split('\n')):
            if line in ["", " ", "\n", " \n"]:
                continue
            # line = torch.as_tensor(line).to(device)
            inputs = processor(line, voice_preset=voice_preset)
            audio = model.generate(**inputs)
            audio = audio.cpu().numpy()
            #Channels Last
            audio = np.moveaxis(audio, -1, 0)
            all_audio = np.concatenate((all_audio, audio), axis=0)
            all_audio = np.concatenate((all_audio, np.zeros((5, 1))), axis=0)
            logger.info(f"Line {i} read")

    # Save audio as wav and as numpy just in case
    sample_rate = model.generation_config.sample_rate
    np.save(os.path.join("Podcast Generator", "episode_audio", f"{filename}.npy"), all_audio)
    scipy.io.wavfile.write(
        os.path.join("Podcast Generator", "episode_audio", f"{filename}.wav"), 
        rate=sample_rate, data=all_audio.astype(np.float32)
    )
    logger.info("Audio Saved")

