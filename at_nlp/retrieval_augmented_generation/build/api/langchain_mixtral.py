import json
import toml
import os

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.runnables import RunnableConfig
from loguru import logger as log
from typing import Any, Mapping, List
from functools import partial
from requests import request
from pathlib import Path
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult
from langchain_core.outputs import Generation


class Mixtral8x7b(BaseLLM):
    model_name = "mixtral8x7b_instruct"
    streaming: bool = False
    llm_host: str = os.environ.get("LLM_SERVER_HOST", "127.0.0.1")
    llm_port: str = os.environ.get("LLM_SERVER_PORT", "7000")
    llm_dispatch_url: str = f"http://{llm_host}:{llm_port}"
    endpoint: str = "/completion"
    url: str = llm_dispatch_url + endpoint
    headers: Mapping[str, str] = {
        "content-type": "application/json",
        "cache-control": "no-cache",
    }
    dispatch = partial(
        request,
        method="POST",
        url=url,
        headers=headers,
        stream=streaming,
    )
    conf: dict = None

    def __init__(self, streaming=False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.streaming = streaming

        log.info("Loading configuration files...")
        config_path = Path("./config.toml").absolute()
        if config_path.exists():
            conf = toml.load(str(config_path))
            llm_conf = conf["mixtral8x7b"]

            for k in llm_conf.keys():
                conf_k = "MIXTRAL8X7B_" + k.upper().replace("-", "_")
                if conf_k in os.environ:
                    log.info(f"Overriding {k} with {os.environ[conf_k]}")
                    try:
                        env_val = float(os.environ[conf_k])
                        if env_val.is_integer():
                            log.info(f"Converting {env_val} to int")
                            env_val = int(env_val)
                        else:
                            log.info(f"Converting {env_val} to float")
                        llm_conf[k] = env_val
                    except ValueError as e:
                        llm_conf[k] = os.environ[conf_k]

            self.conf = llm_conf
            log.debug(f"llm_host: {self.llm_host}")
            log.debug(f"llm_host_port: {self.llm_port}")
            log.debug(f"llm_api_url: {self.llm_dispatch_url}")
            log.debug(f"Model name: {self.model_name}")
        else:
            log.critical("config.toml not found!")
            exit()

    @property
    def _llm_type(self) -> str:
        return "mixtral8x7b"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "streaming": self.streaming,
        }

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "mixtral8x7b"]

    def sanitize_output(self, output: str) -> str:
        output = output.replace("\n", "")
        return output.strip()

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: List[str] | None = None,
        **kwargs: Any,
    ) -> str:
        output = self._generate([input]).generations[0][0].text
        return self.sanitize_output(output)

    def _generate(
        self,
        prompts: List[str],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        self.conf["prompt"] = prompts
        req_str = json.dumps(self.conf)
        resp = self.dispatch(data=req_str)
        resp = resp.json()["content"]
        resp = Generation(text=resp)
        del self.conf["prompt"]
        return LLMResult(generations=[[resp]])
