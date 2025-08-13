import abc
import json
import httpx
from logger_factory import get_logger
from utils import RecoverableError, ModelError


class ILanguageModel(abc.ABC):
    @abc.abstractmethod
    async def generate_response(self, prompt, table_name: str) -> str:
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

    async def query_llm(self, prompt: str, table_name: str) -> str:
        llm_req_logger = get_logger(table_name, "llm-requests")
        error_logger = get_logger(table_name, "error")

        llm_req_logger.debug(f"LLM Request Prompt:\n{prompt}")

        try:
            if self._strategy is None:
                raise ValueError("[query_llm] - No strategy set")
            if not prompt.strip():
                raise ValueError("Prompt cannot be empty")

            return await self._strategy.generate_response(prompt, table_name)
        except Exception as e:
            error_logger.error(f"[query_llm] - Failed to retrieve response: {e}")
            raise ModelError(
                f"[query_llm] - Failed to retrieve response from model: {e}"
            ) from e

    def get_current_llm_name(self) -> str:
        return self._strategy.get_model_name()


class LocalOllamaStrategy(ILanguageModel):
    def __init__(self, args: dict, max_connections):
        self._args = args
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
        timeout = httpx.Timeout(connect=120, read=120.0, write=60.0, pool=None)
        self._client = httpx.AsyncClient(
            base_url=self._args["base_url"],
            headers={"Content-Type": "application/json"},
            limits=limits,
            timeout=timeout,
        )

    async def generate_response(self, prompt: str, table_name: str) -> str:
        llm_resp_logger = get_logger(table_name, "llm-responses")
        error_logger = get_logger(table_name, "error")  # unified name
        url = "/generate"
        payload = {
            "model": self._args["model"],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._args["temperature"],
                # "num_predict": self._args["max_tokens"],
                # "top_p": self._args["top_p"],
            },
        }

        try:
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if "response" not in data:
                raise ModelError(f"No 'response' key in reply: {data}")
            llm_resp_logger.debug(f"LLM Response:\n{data['response']}")
            return data["response"]

        except httpx.ReadTimeout as e:
            error_logger.error(f"LLM request timed out: {e}")
            raise RecoverableError("LLM request timed out") from e
        except httpx.ConnectError as e:
            error_logger.error(f"Could not connect to Ollama: {e}")
            raise ModelError(f"Could not connect to Ollama at {resp.url}") from e
        except httpx.HTTPStatusError as e:
            error_logger.error(
                f"Status error {e.response.status_code}: {e.response.text}"
            )
            raise ModelError(
                f"Status error {e.response.status_code}: {e.response.text}"
            ) from e
        except json.JSONDecodeError as e:
            error_logger.error(f"Invalid JSON in response: {resp.text!r}")
            raise ModelError(f"Invalid JSON in response: {resp.text!r}") from e
        except Exception as e:
            error_logger.error(f"Unexpected error: {e}")
            raise ModelError(f"Unexpected error: {e}") from e

    def get_model_name(self) -> str:
        return self._args["model"]


class VLLMStrategy(ILanguageModel):
    def __init__(self, args: dict, max_connections):
        self._args = args
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
        timeout = httpx.Timeout(connect=120, read=120.0, write=60.0, pool=None)
        self._client = httpx.AsyncClient(
            base_url=self._args["base_url"],
            headers={"Content-Type": "application/json"},
            limits=limits,
            timeout=timeout,
        )

    async def generate_response(self, prompt: str, table_name: str) -> str:
        llm_resp_logger = get_logger(table_name, "llm-responses")
        error_logger = get_logger(table_name, "error")  # unified name
        url = "/v1/completions"
        payload = {
            "prompt": prompt,
            # "max_tokens": 4000,
            # "temperature": 0.2,
        }

        try:
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            if "choices" not in data or not data["choices"] or "text" not in data["choices"][0]:
                raise ModelError(f"Invalid LLM response format: {data}")

            text = data["choices"][0]["text"]

            if not text.strip():
                llm_resp_logger.debug("LLM Response: <EMPTY>")
            else:
                llm_resp_logger.debug(f"LLM Response:\n{text}")
            llm_resp_logger.debug(f"JSON: {json.dumps(data, indent=2)}")
            return text

        except httpx.ReadTimeout as e:
            error_logger.error(f"LLM request timed out: {e}")
            raise RecoverableError("LLM request timed out") from e
        except httpx.ConnectError as e:
            error_logger.error(f"Could not connect to VLLM: {e}")
            raise ModelError(f"Could not connect to VLLM at {resp.url}") from e
        except httpx.HTTPStatusError as e:
            error_logger.error(
                f"Status error {e.response.status_code}: {e.response.text}"
            )
            raise ModelError(
                f"Status error {e.response.status_code}: {e.response.text}"
            ) from e
        except json.JSONDecodeError as e:
            error_logger.error(f"Invalid JSON in response: {resp.text!r}")
            raise ModelError(f"Invalid JSON in response: {resp.text!r}") from e
        except Exception as e:
            error_logger.error(f"Unexpected error: {e}")
            raise ModelError(f"Unexpected error: {e}") from e

    def get_model_name(self) -> str:
        return self._args["model"]