from pathlib import Path

import logging

from utils import RecoverableError

logger = logging.getLogger(__name__)

def get_prompt(prompt_file_name: str, prompts_dir: str = "./prompts") -> str:

    file = Path(prompts_dir) / f"{prompt_file_name}.txt"
    
    if not file.exists():
        raise FileNotFoundError
    
    with open(file, 'r', encoding='utf-8') as file:
        prompt_content = file.read()
    
    prompt_content = prompt_content.rstrip()
    
    if not prompt_content:
        raise ValueError(f"Prompt file '{prompt_file_name}' is empty")
        
    return prompt_content


def build_prompt_for_dynamic_plan(table, question, chain) -> str:
        
    try:
        prompt: str = get_prompt("dynamic_plan")

    except Exception as e:
        raise

    prompt += f"\n{table}\n"
    prompt += "Question: " + question + "\n"
    prompt += "Function Chain: " + str(chain) + "\n"
    prompt += "======================================= Completion ======================================="

    return prompt


def build_prompt_for_generate_args(table, question, f):

    try:
        match f:
            case "f_add_column":
                pattern = f"{f}\\((.*)\\)\\. The value: (.*)\\."
            case "f_select_column":
                pattern = f"{f}\\(\\[(.*)\\]\\)"
            case "f_select_row":
                pattern = f"{f}\\(\\[(.*)\\]\\)"
            case "f_sort_by":
                pattern = f'{f}\\((.*)\\),\\s*the order is "(.*)"'
            case "f_group_by":
                pattern = f"{f}\\((.*)\\)"
            case _:
                raise ValueError(f"received unknown operation: {f}")

        prompt: str = get_prompt(f)
    except ValueError as e:
        raise RecoverableError from e
    except Exception as e:
        raise
    
    prompt += table + "\n"
    prompt += "Question: " + question + "\n"
    prompt += "======================================= Completion ======================================="  

    return prompt, pattern


def get_prompt_for_query(table, question):
    try:
        prompt: str = get_prompt("query")
    except Exception as e:
        raise  

    prompt += "Please answer the following question about the table: \n"
    prompt += f"{table}\n"
    prompt += "Question: " + question + "\n"
    prompt += "======================================= Completion ======================================="
    
    return prompt
