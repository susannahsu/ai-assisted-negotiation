import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

key = os.getenv('OPENAI_API_KEY')
if key is None:
    raise ValueError("The OPENAI_API_KEY environment variable is not set \
                     or .env file is missing.")

client = OpenAI(
    api_key=key
)

user_prompt_path = '../prompts/hfa_prompt.txt'
agent_prompt_path = '../prompts/iceberg_css_prompt.txt'

def load_prompt_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def generate_user_prompt(file_path):
    return load_prompt_from_file(file_path)

def generate_agent_prompt(file_path):
    return load_prompt_from_file(file_path)

def save_gpt_response_to_file(response, run_number, 
                              output_folder="gpt_responses", 
                              output_prefix="gpt_response"):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Structured filename with run number
    output_file_path = os.path.join(output_folder, f'{output_prefix}_{run_number}.txt')

    # Write the response to the text file
    with open(output_file_path, 'w') as file:
        file.write(response)

def run_gpt_process(n_runs=1, model="gpt-4-0125-preview",
                    output_folder="gpt_responses",
                    output_prefix="gpt_response"):
    for run_number in range (1, n_runs + 1):
        agent_prompt = generate_agent_prompt(agent_prompt_path)
        user_input = generate_user_prompt(user_prompt_path)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": agent_prompt},
                {"role": "user", "content": user_input},
            ],
        )
        response_content = completion.choices[0].message.content
        save_gpt_response_to_file(response_content, run_number, output_folder, output_prefix)

N = 30
model = "gpt-4-0125-preview"
output_folder = "../gpt_responses/iceberg_css_responses/hfa"
output_prefix = "iceberg_css_response_hfa"
run_gpt_process(N, model, output_folder, output_prefix)