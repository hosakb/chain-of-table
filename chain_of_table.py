import os
import re
import logging

from dotenv import load_dotenv
import pandas as pd
from datasets import load_from_disk

from model import LLM, LocalOllamaStrategy, ChatGPTStrategy
from table import Table, PandasStrategy
from utils import get_prompt

# TODO: impl retry
# TODO: Improve scoring 
# TODO: recover?


class ChainOfTable:

    def __init__(self, model, table_handler: Table):
        self._model: LLM = model
        self._table_handler: Table = table_handler
        self.score = 0
        self._logger = logging.getLogger(__name__)

    def execute(self, dataset, mock=False):

        for i, table_data in enumerate(dataset):

            question = table_data["question"]
            answer = table_data["answers"]
            self._table_handler.load_from_json(table_data)
            self._logger.info(f"[execute] - Asking LLM:\n{question}\n\nOn table: {self._table_handler.get_caption()}")
            chain = "<Begin> -> "

            while True:
                try:
                    txt_table = self._table_handler.to_str()

                    operation = self._dynamic_plan(txt_table, question, chain)

                    if operation == "<End>":
                        break

                    chain += operation
                    operation = operation.split("(")[0]

                    args = self._generate_args(txt_table, question, operation)


                    self._logger.debug(f"Before Operation\n{self._table_handler.to_str()}")
                    self._table_handler.perform_operation(operation, args)
                    self._logger.debug(f"After Operation\n{self._table_handler.to_str()}")
                    return "DONE"
                
                except Exception as e:
                    self._logger.error(f"[execute] - Failed to complete chain of tables.")
                    raise

            try:
                response = self._query(question)
            except Exception as e:
                    self._logger.error(f"[execute] - Failed to query final table.")
                    raise
            
            if response == answer: # TODO: Improve scoring 
                self.score += 1
                self._logger.error(f"[execute] - Current score: {self.score}")


    def _dynamic_plan(self, table, question, chain) -> str:
        self._logger.info(f"[_dynamic_plan] - fetching next operation")
        try:
            self._logger.debug("[_dynamic_plan] - fetching prompt from file")
            prompt: str = get_prompt("dynamic_plan")

        except Exception as e:
                self._logger.error(f"[_dynamic_plan] - failed to fetch prompt. Error:\n{e}") # TODO: recover?
                raise
        

        prompt += f"\n{table}\n"
        prompt += 'Question: ' + question + '\n'
        prompt += 'Function Chain: ' + str(chain) + '\n'
        prompt += 'Completion: '        
        self._logger.info(f"[_dynamic_plan] - prompting llm")
        
        try:
            
            f = self._model.query_llm(prompt)
            
        except Exception as e:
            self._logger.error(f"[_dynamic_plan] - Failed to get next operation. Error:\n{e}")
            raise

        f_split = f.split(" -> ")[0]
        self._logger.debug(f"[_dynamic_plan] - return value {f}")
        return f_split
            

    def _generate_args(self, table, question, f: str):
        self._logger.info(f"[_generate_args] - fetching args for operation {f}")  

        try:
            match f:
                case "f_add_column":
                    prompt = get_prompt(f)
                    pattern = f"{f}\\((.*)\\). The value: (.*)"
                case "f_select_column":
                    prompt = get_prompt(f)
                    pattern = f"{f}\\(\\[(.*)\\]\\)"
                case "f_select_row":
                    prompt = get_prompt(f)
                    pattern = f"{f}\\(\\[(.*)\\]\\)"
                case "f_sort_by":
                    prompt = get_prompt(f)
                    pattern = f'{f}\\((.*)\\),\\s*the order is "(.*)"\\.'
                case "f_group_by":
                    prompt = get_prompt(f)
                    pattern = f"{f}\\((.*)\\)"
                    
                case _:
                    self._logger.error(f"[_generate_args] - received unknown operation: {f}")
                    raise ValueError(f"received unknown operation: {f}") # TODO: impl retry of _dynamic_plan
                   
        except Exception as e:
                self._logger.error(f"[_generate_args] - failed to fetch prompt for operation{f}. Error:\n{e}") # TODO: recover?
                raise
        
        prompt += table + "\n"
        prompt += 'Question: ' + question + '\n'
        prompt += '======================================= Completion ======================================='
        
        try:
            response = self._model.query_llm(prompt)
        except Exception as e:
                self._logger.error(f"[_generate_args] - failed to prompt llm. Error:\n{e}") # TODO: recover?
                raise

        try:
            match = re.search(pattern, response)
            if not match:
                self._logger.error(f"[_generate_args] - No arguments found in llm response: {response}.") 
                raise ValueError(f"No arguments found for {f} in llm response: {response}.") # TODO: recover?
        except Exception as e:
            self._logger.error(f"[_generate_args] - Failed to match arguments for operation {f}. Error:\n{e}") 
            raise # TODO: recover?

        self._logger.debug(f"[_generate_args] - return value {match}")
        return match

    def _query(self, question):
        self._logger.info(f"[_query] - Querying final table")
        try:
            table = self._table_handler.to_str()
        except Exception as e:
            self._logger.error(f"[_query] - failed to fetch string representation of table. Error:\n{e}") 
            raise # TODO: recover?

        try:
            prompt: str = get_prompt("query")
        except Exception as e:
            self._logger.error(f"[_query] - failed to fetch prompt for query. Error:\n{e}") 
            raise # TODO: recover?
        
        prompt += f"{table}\n"
        prompt += 'Question: ' + question + '\n'
        prompt += '======================================= Completion ======================================='
        self._logger.debug(f"[_query] - complete prompt:\n{prompt}")
        
        try:
            response = self._model.query_llm(prompt)
        except Exception as e:
                self._logger.error(f"[_query] - failed to prompt llm. Error:\n{e}") 
                raise # TODO: recover?

        return response


def __main__():
    load_dotenv()

    logger =  logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logs_dir = "./logs/"
    os.makedirs(logs_dir, exist_ok=True)
    console_handler = logging.FileHandler("./logs/logs.log", mode="w")
    logger.addHandler(console_handler)


        # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        # if OPENAI_API_KEY is None:
        #     return "Missing API Key"

        # config = {
        #     "api_key": OPENAI_API_KEY,
        #     "model": "gpt-3",
        #     "temperature": 0.7,
        #     "max_tokens": 64,
        #     "top_p": 1,
        #     "frequency_penalty": 0,
        #     "presence_penalty": 0,
        # }

        # model_strategy = ChatGPTStrategy(config)
    print("Execute")
    base_url = os.getenv("OLLAMA_BASE_URL")

    config = {
        "base_url": base_url,
        "model": "llama3:latest",
        "temperature": 0.7,
        "max_tokens": 64,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    

    model_strategy = LocalOllamaStrategy(config)
    model = LLM(model_strategy)

    table_strategy = PandasStrategy()
    table_handler = Table(table_strategy)

    local_dataset_path = "./wikitablequestions_parquet_store"
    os.makedirs(local_dataset_path, exist_ok=True)
    
    dataset = load_from_disk(local_dataset_path)["train"]
    
    cot = ChainOfTable(model, table_handler)
    cot.execute(dataset)


__main__()