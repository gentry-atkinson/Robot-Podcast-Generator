# Robot Podcast Generator
This project will generate one episode a podcast called "No Humans Were Involved with
this Podcast" when run. The structure of each episode will be as follows:

- Introduction
- (4 segments chosen from 7 possible)
  - AI and Pop Culture
  - AI Fails
  - AI Q&A
  - Future Forecast
  - Listener Stories
  - Tech Tales
  - Tech Trivia
- Today's Sponsor
- Conclusion

Each segment is generated using a separate prompt. The prompts for this system are
(currently) human generated, but all of the episode content is written and recorded
by a generative AI. The episode currently run 20-30 minutes in length and are generated
as .wav files. Additionaly, the script for each episode is recorded as a .txt file.

## Models Used
All of the generative models being used in this project are provided by Hugging Face.
The list of models used is:

- Zephyr (finetuned from Google's Gemma 7b): text generation
- Suno's Bark: voice generation from text
- Facebook's Musicgen Small: intro and transition music

## Using This Project
This project is under heavy development and you WILL experience difficulty running
it yourself. You may find yourself having to build an appropriate directory tree
and tweaking a line or two.

Having said that:

  python3 main.py

...is all you need to run. The main file will run the others as necesary.

Anecdotally, this program takes about 12 hours to run on an 8-core CPU with an
RTX 4070. Your milage may vary.

## ToDo
- Outro tune
- Settle on one Bark voice preset
- Eleminate Bark hallucination
- More variety in titles
- Replace Gemma 7b with a newer model (MPT 30b?)
- More eloquent line splitting
- Generate visual art for show
- Review sound quality (reported "ringing")
- RSS feed