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

import time

ppe_prompts = constants.ppe_prompts

def find_project_root(current_path: str | Path | None = None) -> Path:
    """Finds the project root directory by checking for common project files.

    Args:
        current_path: The starting path (defaults to the current working directory).

    Returns:
        The Path object of the project root, or Path('.') if not found.
    """
    project_files = [".git", "setup.py", "requirements.txt", "src"]
    if current_path is None:
        current_path = Path.cwd()
    elif isinstance(current_path, str):
        current_path = Path(current_path)

    if any((current_path / file).exists() for file in project_files):
        return current_path
    for parent in current_path.parents:
        if any((parent / file).exists() for file in project_files):
            return parent
    return Path()  # Return . instead of Path() for clarity

data_directory = find_project_root() / 'data'
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


if __name__ == "__main__":
    _output_csv = sys.argv[1]

    prompt_text = input("Enter your prompt: ")

    if len(prompt_text) == 0:
        default_prompts(_output_csv)
    else:
        custom_prompt(prompt_text)



