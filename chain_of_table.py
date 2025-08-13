import os
import re
import logging
import asyncio
import shutil

from dotenv import load_dotenv
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

from logger_factory import get_logger
from model import LLM, LocalOllamaStrategy, VLLMStrategy
from table import Table, PandasStrategy
from utils import ModelError, TableError, RecoverableError
from prompt import (
    build_prompt_for_dynamic_plan,
    build_prompt_for_generate_args,
    get_prompt_for_query,
)

MAX_CONCURRENT = 256
MAX_CLIENTS = 256


class ChainOfTable:
    def __init__(self, model, table_handler: Table):
        self._model: LLM = model
        self._table_handler: Table = table_handler
        self._score = 0

    async def execute(self, dataset, max_concurrent=16):
        queue = asyncio.Queue()

        for table_data in dataset:
            await queue.put(table_data)

        pbar = tqdm(total=len(dataset), desc="Processing tables", unit="table")

        async def worker():
            while not queue.empty():
                table_data = await queue.get()
                try:
                    await self._process_singel_table(table_data)
                finally:
                    queue.task_done()
                    pbar.update(1)  
                    
        tasks = [asyncio.create_task(worker()) for _ in range(max_concurrent)]
        await queue.join()  # Wait until all tasks are processed
        for t in tasks:
            t.cancel()

        pbar.close()
        return self._score

    async def _process_singel_table(self, table_data):
        MAX_CHAIN_LENGTH = 4

        question = table_data["question"]
        answer = table_data["answers"][0]
        table_name = table_data["table"]["name"].replace(" ", "_")

        logic_logger = get_logger(table_name, "chain-of-tables")
        error_logger = get_logger(table_name, "error")

        self._table_handler.load_from_json(table_data)
        chain = "<Begin> -> "
        debug_chain = "<Begin> -> "
        chain_length = 0

        while chain_length <= MAX_CHAIN_LENGTH:
            txt_table = self._table_handler.to_str()

            try:
                operation = await self._dynamic_plan(
                    txt_table, question, chain, table_name
                )
                debug_chain += "_dynamic_plan -> "
                if operation == "<End>":
                    break

                operation_name = operation.split("(")[0]
                args = await self._generate_args(
                    txt_table, question, operation_name, table_name
                )
                debug_chain += "_generate_args -> "
                logic_logger.warning(f"args: {args}")
                self._table_handler.perform_operation(
                    operation_name, args, table_name
                )
                debug_chain += "perform_operation -> "
                chain += operation
                chain_length = len(chain.split("->")) - 1

            except RecoverableError as e:
                error_logger.warning(f"Recoverable error: {e}")
                debug_chain += "error -> "
                break
            except Exception as e:
                error_logger.exception(f"Unexpected error: {e}")
                debug_chain += "error -> "
                raise

        try:
            response = await self._query(question, table_name)
            debug_chain += "_query"
        except RecoverableError as e:
            error_logger.warning(f"Recoverable error during query: {e}")
            response = "FAILED ATTEMPT"
        except Exception as e:
            error_logger.exception(f"Unexpected error during query: {e}")
            raise

        logic_logger.debug(f"Table: {table_name}\n\n-Debug chain: {debug_chain}\n\nFunction Chain: {chain}\n\nQuestion: {question}\n\nGenerated Answer: {response}\n\nActual Answer: {answer}")
        if response == answer:
            self._score += 1
            logic_logger.info(
                f"[execute] - ✅ Correct for {self._table_handler.get_caption()}"
            )
        else:
            logic_logger.info(
                f"[execute] - ❌ Incorrect for {self._table_handler.get_caption()}"
            )

    async def _dynamic_plan(self, table, question, chain, table_name: str) -> str:
        prompt_logger = get_logger(table_name, "prompts")
        llm_resp_logger = get_logger(table_name, "llm-responses")

        prompt = build_prompt_for_dynamic_plan(table, question, chain)
        prompt_logger.debug(f"[dynamic_plan] Prompt:\n{prompt}")

        try:
            response = await self._model.query_llm(prompt, table_name)
            llm_resp_logger.debug(f"[dynamic_plan] Response:\n{response}")

            pattern = re.compile(r"f_.*?<END>")
            matches = pattern.findall(response)[0]
        except IndexError as e:
            raise RecoverableError from e
        return matches.split(" -> ")[0]

    async def _generate_args(self, table, question, f: str, table_name: str):
        prompt_logger = get_logger(table_name, "prompts")
        llm_resp_logger = get_logger(table_name, "llm-responses")
        logic_logger = get_logger(table_name, "generate_args")

        prompt, pattern = build_prompt_for_generate_args(table, question, f)
        prompt_logger.debug(f"[generate_args] Prompt:\n{prompt}")

        response = await self._model.query_llm(prompt, table_name)
        llm_resp_logger.debug(f"[generate_args] Response:\n{response}")
        logic_logger.debug(f"[generate_args] Response:\n{response}")
        match = re.search(pattern, response)
        logic_logger.debug(f"[generate_args] match:\n{match}")
        if not match:
            raise ValueError(
                f"No arguments found for {f} in llm response: {response}."
            )
        return match

    async def _query(self, question, table_name: str):
        prompt_logger = get_logger(table_name, "prompts")
        llm_resp_logger = get_logger(table_name, "llm-responses")

        table = self._table_handler.to_str()
        prompt = get_prompt_for_query(table, question)
        prompt_logger.debug(f"[query] Prompt:\n{prompt}")

        response = await self._model.query_llm(prompt, table_name)
        llm_resp_logger.debug(f"[query] LLM Response:\n{response}")
       
        match = re.search(r"The Answer is:\s*(.*?)(?=\.|\n)",  response, re.IGNORECASE)
        if match:
            llm_resp_logger.debug(f"[query] LLM match:\n{match.group(1)}")
            return match.group(1)
        raise RecoverableError(f"[query] Failed to match pattern in query Response.\nResponse: {response}")


def __main__():
    load_dotenv()

    logs_dir = "./logs"
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)
    os.makedirs(logs_dir, exist_ok=True)

    logger = get_logger("GLOBAL", "main")
    error_logger = get_logger("GLOBAL", "error")
    logger.setLevel(logging.DEBUG)

    logs_dir = "./logs/"
    os.makedirs(logs_dir, exist_ok=True)

    local_dataset_path = "./wikitablequestions_parquet_store"

    # base_url = os.getenv("OLLAMA_BASE_URL")
    base_url = os.getenv("VLLM_BASE_URL")
    config = {
        "base_url": base_url,
        "model": "llama2:13b-chat",
        "temperature": 0.7,
        "max_tokens": 5000,
        "top_p": 0.8,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 0.8,
        "stop": [...]
    }

    # model_strategy = LocalOllamaStrategy(config)
    model_strategy = VLLMStrategy(config, MAX_CLIENTS)
    accuracy = None

    try:
        model = LLM(model_strategy)
        table_strategy = PandasStrategy()
        table_handler = Table(table_strategy)

        os.makedirs(local_dataset_path, exist_ok=True)
        dataset = load_from_disk(local_dataset_path)["train"]
        dataset = dataset.select(range(0, 1000))

        score = asyncio.run(
        ChainOfTable(model, table_handler).execute(dataset, MAX_CONCURRENT)
    )
        accuracy = score / len(dataset)
        print(f"Accuracy: {accuracy:.2%}")

    except (ModelError, TableError) as e:
        error_logger.exception(f"Critical error: {e}")
    except Exception as e:
        error_logger.exception(f"Unexpected error: {e}")

    if accuracy is not None:
        print(f"accuracy: {accuracy}")
    else:
        print("No accuracy could be computed due to an error.")


__main__()