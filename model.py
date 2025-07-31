import abc
import requests
import json
import logging

import openai
import httpx
from utils import RecoverableError, ModelError


class ILanguageModel(abc.ABC):
    def __init__(self):
        self.logger = logging.getLogger("my_logger")

    @abc.abstractmethod
    async def generate_response(self, prompt) -> str:
        pass

    @abc.abstractmethod
    def get_model_name(self) -> str:
        pass


class LLM:
    _strategy: ILanguageModel

    def __init__(self, strategy: ILanguageModel):
        if not isinstance(strategy, ILanguageModel):
            raise TypeError("Provided strategy must be an instance of LLMStrategy.")
        self._strategy = strategy

    def set_strategy(self, strategy: ILanguageModel):
        if not isinstance(strategy, ILanguageModel):
            raise TypeError("Provided strategy must be an instance of LLMStrategy.")
        self._strategy = strategy

    async def query_llm(self, prompt: str) -> str:
        try:
            if self._strategy is None:
                raise ValueError("[query_llm] - No strategy set")

            if not prompt or not prompt.strip():
                raise ValueError("Prompt cannot be empty")

            return await self._strategy.generate_response(prompt)
        except Exception as e:
            raise ModelError(
                f"[query_llm] - Failed to retrieve response from model: {e}"
            ) from e

    def get_current_llm_name(self) -> str:
        return self._strategy.get_model_name()


########################################## Strategies #################################################


class ChatGPTStrategy(ILanguageModel):
    def __init__(self, args):
        self._api_key = args["api_key"]
        self._args = args

    async def generate_response(self, prompt: str) -> str:

        openai.api_key = self._api_key
        try:
            response = await openai.chat.completions.create(
                model=self._args["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self._args["temperature"],
                max_tokens=self._args["max_tokens"],
                top_p=self._args["top_p"],
                frequency_penalty=self._args["frequency_penalty"],
                presence_penalty=self._args["presence_penalty"],
            )

            content = response.choices[0].message.content

            if content is None:
                raise ValueError("Received empty response from OpenAI")

            return content.strip()

        except openai.APIError as e:
            raise ModelError("[generate_response] - OpenAI API error occurred.") from e
        except Exception as e:
            raise ModelError(
                f"[generate_response] - Unexpected error in response generation: {e}"
            ) from e

    def get_model_name(self) -> str:
        return f"ChatGPT-{self._args['model']}"


class LocalOllamaStrategy(ILanguageModel):
    def __init__(self, args: dict):
        super().__init__()
        self._args = args

        limits = httpx.Limits(
            max_connections=24,
            max_keepalive_connections=24,
        )

        timeout = httpx.Timeout(connect=120, read=120.0, write=60.0, pool=None)

        self._client = httpx.AsyncClient(
            base_url=self._args["base_url"],
            headers={"Content-Type": "application/json"},
            limits=limits,
            timeout=timeout
        )

    async def generate_response(self, prompt: str) -> str:

        url = "/generate"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self._args["model"],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._args["temperature"],
                "num_predict": self._args["max_tokens"],
                "top_p": self._args["top_p"],
            },
        }

        try:
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()

            data = resp.json()
            if "response" not in data:
                raise ModelError(
                    f"[generate_response] - no “response” key in reply: {data}"
                )
            self.logger.info(f"Response: {data['response']}")
            return data["response"]
        
        except httpx.ReadTimeout as e:
            self.logger.warning(f"[generate_response] - read timeout ({self._client.timeout.read}s)")
            raise RecoverableError("LLM request timed out") from e

        except httpx.ConnectError as e:
            raise ModelError(
                f"[generate_response] - could not connect to Ollama at {resp.url}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ModelError(
                f"[generate_response] - status error {e.response.status_code}: "
                f"{e.response.text}"
            ) from e
        except json.JSONDecodeError as e:
            raise ModelError(
                f"[generate_response] - invalid JSON in response: {resp.text!r}"
            ) from e
        except Exception as e:
            self.logger.exception(f"[generate_response] - unexpected error: {e}")
            raise ModelError(f"[generate_response] - unexpected error: {e}") from e

    def get_model_name(self) -> str:
        return self._args["model"]
