import base64
import csv
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.tools import calculate_performance
from src.ppe_prompt_benchmarking import constants
import os
import openai
import numpy as np
import cv2
import time

ppe_prompts = constants.ppe_prompts
data_directory = Path('/Users/john.huynh/Personal/Projects/PPEPromptBenchmarking/data')
frames_directory = data_directory / 'frames'
def default_prompts(output_csv: str):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

        image_paths = [file for file in frames_directory.iterdir() if file.suffix.lower() in image_extensions]

        for key, prompt_text in ppe_prompts.items():
            print(key, "\n")
            for img_path in image_paths:
                send_prompt_and_frame(prompt_text, img_path)
def custom_prompt(prompt_text: str):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    image_paths = [file for file in frames_directory.iterdir() if file.suffix.lower() in image_extensions]

    for img_path in image_paths:
        send_prompt_and_frame(prompt_text, img_path)
def send_prompt_and_frame(prompt_text: str , frame_path: Path):
    filename = frame_path.name
    print(f"Frame {filename} \n")
    start_time = time.perf_counter()
    completion = openai_query(prompt_text, frame_path)
    print(completion.choices[0].message.content)
    duration = time.perf_counter() - start_time
    print(f"Prompt took"
          f" {duration:.2f}s with Frame {filename} \n")
def openai_query(prompt_text: str, frame_path: Path):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI()

    base64_image = encode_image(frame_path)

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user",
             "content": [
                 {
                     "type": "text",
                     "text": prompt_text
                 },
                 {
                     "type": "image_url",
                     "image_url": {
                         "url": f"data:image/jpeg;base64,{base64_image}"
                     }
                 }
             ]
             },
        ],
        max_tokens= 300
    )
    return completion
def encode_image(image_path):
    """Encodes an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_boolean_input(prompt):
    while True:
        user_input = input(prompt).lower()
        if user_input == "yes":
            return True
        elif user_input == "false":
            return False
        else:
            print("Invalid input. Please enter 'Yes' or 'No'.")

if __name__ == "__main__":
    _output_csv = sys.argv[1]

    prompt_text = input("Enter your prompt: ")

    if len(prompt_text) == 0:
        default_prompts(_output_csv)
    else:
        custom_prompt(prompt_text)



