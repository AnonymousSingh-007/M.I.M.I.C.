import pyttsx3
import random
from typing import List, Any

def speak_anime_line():
    engine = pyttsx3.init()
    voices: List[Any] = engine.getProperty('voices')  # type: ignore
    if voices:
        engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 180)

    lines = [
        "Nyaa~ cursor on the move, senpai!",
        "Hehe, I'm tracing your path desu!",
        "Sugoi~ I feel like a hacker uchiha now!",
        "Mangekyo Sharingan activated, watch me go!",
        "The world is imperfect, but the mouse path is flawless!"
    ]
    line = random.choice(lines)
    engine.say(line)
    engine.runAndWait()
