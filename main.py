# Author: Gentry Atkinson
# Organization: St. Edwards University
# Date: 07 Mar, 2024


from transformers import pipeline, AutoProcessor, BarkModel
import torch
import os
from datetime import datetime
import gc
from datasets import load_dataset
import numpy as np
import random
import scipy

if __name__ == "__main__":
    # Setup
    USE_CPU = False

    if torch.cuda.is_available() and not USE_CPU:
        device = "cuda"
    else:
        device = "cpu"

    input_text = ""
    filename = f"episode_{datetime.now()}"
    title_prompt = "Generate a title for one episode of a podcast that is named 'No Humans Were Involved with This Podcast'. This podcast is about AI and society. The title should be fun and witty."
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
    print(f"Episode title: {title}")

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
        print(f"{segment} generated. {len(script_segment)} characters")
        script[segment] = script_segment

    with open(os.path.join("Podcast Generator", "scripts", f"{filename}.txt"), 'w+') as f:
        f.write(script)
    print(f"Script finished. Total length: {len(script.split(' '))} words")


    #Clean up models for memory

    del pipe
    del outputs
    del messages
    gc.collect()

    # Convert Script to Audio

    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")
    # model.to(device)
    # processor.to(device)
    voice_preset = "v2/en_speaker_6"
    all_audio = np.zeros((200, 1))

    for title, text in script.items():
        if title != "Introduction":
            all_audio = np.concatenate((np.zeros((50, 1)), audio), axis=0)
            inputs = processor(f"Coming up we have {title}", voice_preset=voice_preset)
            audio = model.generate(**inputs)
            audio = audio.cpu().numpy()
            audio = np.moveaxis(audio, -1, 0)
            all_audio = np.concatenate((all_audio, audio), axis=0)
            all_audio = np.concatenate((np.zeros((5, 1)), audio), axis=0)
        for i, line in enumerate(script[text].split('\n')):
            if line in ["", " ", "\n", " \n"]:
                continue
            # line = torch.as_tensor(line).to(device)
            inputs = processor(line, voice_preset=voice_preset)
            audio = model.generate(**inputs)
            audio = audio.cpu().numpy()
            #Channels Last
            audio = np.moveaxis(audio, -1, 0)
            all_audio = np.concatenate((all_audio, audio), axis=0)
            all_audio = np.concatenate((np.zeros((5, 1)), audio), axis=0)
            print(f"Line {i} read")

    # Save audio as wav and as numpy just in case
    sample_rate = model.generation_config.sample_rate
    np.save(os.path.join("Podcast Generator", "episode_audio", f"{filename}.npy"), all_audio)
    scipy.io.wavfile.write(
        os.path.join("Podcast Generator", "episode_audio", f"{filename}.wav"), 
        rate=sample_rate, data=all_audio.astype(np.float32)
    )
    print("Audio Saved")

