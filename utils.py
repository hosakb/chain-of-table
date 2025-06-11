import logging
from pathlib import Path

def get_prompt(prompt_name: str, prompts_dir: str = "./prompts") -> str:
    logger = logging.getLogger(__name__)
    
    try:
        prompt_file = Path(prompts_dir) / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            logger.error(f"[get_prompt] - Prompt file not found: {prompt_file}")
            raise FileNotFoundError(f"Prompt '{prompt_name}' not found at {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as file:
            prompt_content = file.read()
        
        prompt_content = prompt_content.rstrip()
        
        if not prompt_content:
            logger.warning(f"[get_prompt] - Prompt file '{prompt_name}' is empty")
            raise ValueError(f"Prompt file '{prompt_name}' is empty")
        
        return prompt_content
        
    except FileNotFoundError:
        raise 
    except Exception as e:
        logger.error(f"[get_prompt] - Failed to read prompt '{prompt_name}': {e}")
        raise