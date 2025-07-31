import os
import re
import logging
import asyncio

from dotenv import load_dotenv
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

from model import LLM, LocalOllamaStrategy, ChatGPTStrategy
from table import Table, PandasStrategy
from utils import ModelError, TableError, RecoverableError
from prompt import  build_prompt_for_dynamic_plan, build_prompt_for_generate_args, get_prompt_for_query

MAX_CONCURRENT = 1


class ChainOfTable:

    def __init__(self, model, table_handler: Table):
        self._model: LLM = model
        self._table_handler: Table = table_handler
        self._score = 0
        self._logger = logging.getLogger("my_logger")
        
    async def execute(self, dataset, max_concurrent=1):

        sem = asyncio.Semaphore(max_concurrent)

        tasks = []
        async def worker(row):
            async with sem:
                return await self._process_singel_table(row)

        for table_data in tqdm(dataset):
            tasks.append(asyncio.create_task(worker(table_data)))

        for finished in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                await finished

        return self._score
    
    async def _process_singel_table(self, table_data):

        # MAX_RETRIES = 3
        MAX_CHAIN_LENGTH = 4

        question = table_data["question"]
        answer = table_data["answers"]
        self._table_handler.load_from_json(table_data)
        self._logger.info(
            f"[execute] - Asking LLM:\n{question}\nOn {self._table_handler.get_caption()}"
        )
        chain = "<Begin> -> "
        chain_length = 0
        # attempt = 1

        while chain_length <= MAX_CHAIN_LENGTH:
            txt_table = self._table_handler.to_str()
            handler_backup = self._table_handler

            try:
                operation = await self._dynamic_plan(txt_table, question, chain)
                if operation == "<End>":
                    break

                operation_name = operation.split("(")[0]
                args = await self._generate_args(txt_table, question, operation_name)

                self._table_handler.perform_operation(operation_name, args)
                chain += operation
                chain_length = len(chain.split("->")) - 1

            except RecoverableError as e:
                self._table_handler = handler_backup

                # if attempt >= MAX_RETRIES:
                #     break

                # attempt += 1
                response = "FAILED ATTEMPT" # vs. retry
                continue
            except Exception:
                raise

        # for attempt in range(1, MAX_RETRIES + 1):
            
        try:
            response = await self._query(question)
            # break
        except RecoverableError as e:
            
            # if attempt >= MAX_RETRIES:
            response = "FAILED ATTEMPT"
                # break
                
        except Exception as e:
            raise

        
        if response == answer:
            self._score += 1
            self._logger.info(f"[execute] - ✅ for {self._table_handler.get_caption()}")
        else:
            self._logger.info(f"[execute] - answerd x for {self._table_handler.get_caption()}")

    async def _dynamic_plan(self, table, question, chain) -> str:

        prompt = build_prompt_for_dynamic_plan(table, question, chain)

        try:
            response = await self._model.query_llm(prompt)

            pattern = re.compile(r"f_.*?<END>")
            matches = pattern.findall(response)[0]

        except IndexError as e:
            raise RecoverableError from e

        except Exception as e:
            raise

        f_split = matches.split(" -> ")[0]
        return f_split

    async def _generate_args(self, table, question, f: str):
        prompt, pattern = build_prompt_for_generate_args(table, question, f)

        try:
            response = await self._model.query_llm(prompt)
        except Exception as e:
            raise

        try:
            match = re.search(pattern, response)
            if not match:
                raise ValueError(
                    f"No arguments found for {f} in llm response: {response}."
                )
        except Exception as e:
            raise  
        return match

    async def _query(self, question):
        try:
            table = self._table_handler.to_str()
        except Exception as e:
            raise  

        prompt = get_prompt_for_query(table, question)

        try:
            response = await self._model.query_llm(prompt)
        except Exception as e:
            raise
        return response


def __main__():
    load_dotenv()

    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    logs_dir = "./logs/"
    os.makedirs(logs_dir, exist_ok=True)
    console_handler = logging.FileHandler("./logs/logs.log", mode="w")
    logger.addHandler(console_handler)

    base_url = os.getenv("OLLAMA_BASE_URL")
    local_dataset_path = "./wikitablequestions_parquet_store"

    config = {
        "base_url": base_url,
        "model": "llama2:13b-chat",
        "temperature": 0.7,
        "max_tokens": 200,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    model_strategy = LocalOllamaStrategy(config)
    accuracy = None

    try:
        model = LLM(model_strategy)
        table_strategy = PandasStrategy()
        table_handler = Table(table_strategy)

        os.makedirs(local_dataset_path, exist_ok=True)
        dataset = load_from_disk(local_dataset_path)["train"]

        score = asyncio.run(ChainOfTable(model, table_handler).execute(dataset, MAX_CONCURRENT))
        accuracy = score / len(dataset)

    except (ModelError, TableError) as e:
        msg = f"Terminating application due to a critical error: \n{e}"
        logger.exception(msg)
    except Exception as e:
        msg = f"Terminating application due to an unexpected error: \n{e}"
        logger.exception(msg)

    if accuracy is not None:
        print(f"accuracy: {accuracy}")
    else:
        print("No accuracy could be computed due to an error.")


__main__()
