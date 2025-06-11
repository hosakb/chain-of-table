import abc
import requests
import json

import openai

# TODO: impl logging


class ILanguageModel(abc.ABC):
    @abc.abstractmethod
    def generate_response(self, prompt) -> str:
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

    def query_llm(self, prompt: str) -> str:
        if self._strategy is None:
            raise RuntimeError("[query_llm] - No strategy set")

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            return self._strategy.generate_response(prompt)
        except Exception as e:
            # self.logger.error(f"Query failed: {e}")
            raise e

    def get_current_llm_name(self) -> str:
        return self._strategy.get_model_name()  # type: ignore


########################################## Strategies #################################################


class ChatGPTStrategy(ILanguageModel):
    def __init__(self, args):
        self._api_key = args["api_key"]
        self._args = args

    def generate_response(self, prompt: str) -> str:

        # TODO: validate args

        openai.api_key = self._api_key
        try:
            response = openai.chat.completions.create(
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

            # self.logger.debug(f"Generated response length: {len(content)}")
            return content.strip()

        except openai.APIError as e:
            # self.logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {e}")
        except Exception as e:
            # self.logger.error(f"Unexpected error in response generation: {e}")
            raise RuntimeError(f"Failed to generate response: {e}")

    def get_model_name(self) -> str:
        return f"ChatGPT-{self._args['model']}"


class LocalOllamaStrategy(ILanguageModel):
    def __init__(self, args: dict):
        self._args = args

    def generate_response(self, prompt: str) -> str:

        # TODO: validate args

        url = f"{self._args['base_url']}/generate"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self._args["model"],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._args["temperature"],
                "num_predict": self._args["max_tokens"],
                "top_p": self._args["top_p"],
                "frequency_penalty": self._args["frequency_penalty"],
                "presence_penalty": self._args["presence_penalty"],
            },
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            data = response.json()
            if "response" in data:
                return data["response"]
            else:
                raise ValueError(
                    f"'response' key not found in the API response.\nFull response: {data}"
                )

        except requests.exceptions.ConnectionError as e:
            print(f"Error: Could not connect to Ollama at {url}.")
            print("Please ensure Ollama is running and accessible.")
            raise
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the API request: {e}")
            raise
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from response: {response.text}")
            raise

    def get_model_name(self) -> str:
        return self._args["model"]
