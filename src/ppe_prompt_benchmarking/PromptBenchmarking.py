import base64
import csv
import sys
from pathlib import Path

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
    with open("output.txt", 'w', newline='', encoding='utf-8') as file:

        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

        image_paths = [file for file in frames_directory.iterdir() if file.suffix.lower() in image_extensions]
        ground_truth = []
        prediction = []
        compliance_strings = ["PPE_COMPLIANT", "yes", "Yes"]
        noncompliance_strings = ["PPE_NON_COMPLIANT", "no", "No", "Unsure", "Unsure"]
        for key, prompt_text in ppe_prompts.items():
            file.write(f'{key}\n')
            for img_path in image_paths:
                file.write('-----------------------------------------------------------------------------------------------------------------------\n')
                filename = img_path.name
                if "no ppe" in filename.lower():
                    ground_truth.append(0)
                else:
                    ground_truth.append(1)
                filename = img_path.name
                file.write(f'Frame {filename} \n')
                start_time = time.perf_counter()
                completion = send_prompt_and_frame(prompt_text, img_path)
                file.write(f'Result \"{completion}\" \n')
                duration = time.perf_counter() - start_time
                file.write(f"{key} took"
                      f" {duration:.2f}s with Frame {filename} \n")
                file.write('-----------------------------------------------------------------------------------------------------------------------\n')
                if any(text in completion for text in compliance_strings):
                    prediction.append(1)
                elif any(text in completion for text in noncompliance_strings):
                    prediction.append(0)
                else:
                    prediction.append(0)
            cm = calculate_performance.create_confusion_matrix(ground_truth, prediction)
            file.write(f"Confusion matrix:\n {cm}")
            precision = calculate_performance.compute_precision(cm)
            file.write(f"Precision{precision:.2f} \n")
            recall = calculate_performance.compute_recall(cm)
            file.write(f"Recall{recall:.2f} \n")
            f1 = calculate_performance.compute_f1(precision, recall)
            file.write(f"F1-Score{f1:.2f} \n")
def custom_prompt(prompt_text: str):
    with open("output.txt", 'w', newline='', encoding='utf-8') as file:
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

        image_paths = [file for file in frames_directory.iterdir() if file.suffix.lower() in image_extensions]
        ground_truth = []
        prediction = []
        compliance_strings = ["PPE_COMPLIANT", "yes", "Yes"]
        noncompliance_strings = ["PPE_NON_COMPLIANT", "no", "No", "Unsure", "Unsure"]
        for img_path in image_paths:
            print(
                '-----------------------------------------------------------------------------------------------------------------------')
            filename = img_path.name
            if "no ppe" in filename.lower():
                ground_truth.append(0)
            else:
                ground_truth.append(1)
            print(f'Frame {filename}')
            start_time = time.perf_counter()
            completion = send_prompt_and_frame(prompt_text, img_path)
            print(f'Result \"{completion}\"')
            duration = time.perf_counter() - start_time
            print(f"Custom prompt took"
                       f" {duration:.2f}s with Frame {filename}")
            print(
                '----------------------------------------------------------------------------------------------------------------------- \n')
            if any(text in completion for text in compliance_strings):
                prediction.append(1)
            elif any(text in completion for text in noncompliance_strings):
                prediction.append(0)
            else:
                prediction.append(0)
        cm = calculate_performance.create_confusion_matrix(ground_truth, prediction)
        print(f"Confusion matrix:\n {cm}")
        precision = calculate_performance.compute_precision(cm)
        print(f"Precision{precision:.6f}")
        recall = calculate_performance.compute_recall(cm)
        print(f"Recall{recall:.6f}")
        f1 = calculate_performance.compute_f1(precision, recall)
        print(f"F1-Score{f1:.6f}")
def send_prompt_and_frame(prompt_text: str , frame_path: Path):
    completion = openai_query(prompt_text, frame_path)
    return completion.choices[0].message.content
def compute_confusion_matrix(ground_truth: list, prediction: list):
    cm = calculate_performance.create_confusion_matrix(ground_truth, prediction)


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



