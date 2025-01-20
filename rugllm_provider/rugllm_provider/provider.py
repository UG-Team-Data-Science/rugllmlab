# rugllm_provider/provider.py
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
# from jupyter_ai import EnvAuthStrategy, Field
from jupyter_ai_magics import BaseProvider, Persona
from jupyter_ai_magics.providers import EnvAuthStrategy, Field, TextField
from os import getenv

class RugHbLlm(BaseProvider, OpenAI):
    id = "rughbllm"
    name = "RugHbLlm"
    models = [
        "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",
        "mistralai/Mistral-7B-v0.1",
        "google/gemma-2b"
    ]
    help = "Click here for more details on [RugLlm](https://rug.nl)"
    model_id_key = "model_name"
    model_id_label = "Model ID"
    pypi_package_deps = ["langchain_openai"]
    auth_strategy = EnvAuthStrategy(
        name="RUGHB_API_KEY", keyword_param="openai_api_key"
    )
    openai_api_base: str = getenv("RUGHB_API_BASE", 'http://localhost:8000/v1')
    
    fields: ClassVar[List[Field]] = [
        TextField(
            key="openai_api_base",
            label="OpenAI API Base",
            default=getenv("RUGHB_API_BASE", "http://localhost:8000/v1"),
            format="text",
        )
    ]
    
    openai_organization: str = "University of Groningen"
    persona: str = Persona(name="RugHbLlm", avatar_route="api/ai/static/jupyternaut.svg")

    
    def __init__(self, **kwargs):
        # Fetch the openai_api_base from the provided configuration or environment variable
        self.openai_api_base = kwargs.get(
            "openai_api_base",
            getenv("RUGHB_API_BASE", "http://localhost:8000/v1"),
        )
        super().__init__(**kwargs)

class ChatRugHbLlm(BaseProvider, ChatOpenAI):
    id = "rughbllm"
    name = "RugHbLlm"
    models = [
        "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",
        "mistralai/Mistral-7B-v0.1",
        "google/gemma-2b"
    ]
    help = "Click here for more details on [RugLlm](https://rug.nl)"
    model_id_key = "model_name"
    model_id_label = "Model ID"
    pypi_package_deps = ["langchain_openai"]
    auth_strategy = EnvAuthStrategy(
        name="RUGHB_API_KEY", keyword_param="openai_api_key"
    )
    openai_api_base: str = getenv("RUGHBLLM_API_BASE", 'http://localhost:8000/v1')

    fields: ClassVar[List[Field]] = [
        TextField(
            key="openai_api_base",
            label="OpenAI API Base",
            default=getenv("RUGHB_API_BASE", "http://localhost:8000/v1"),
            format="text",
        )
    ]
    
    openai_organization: str = "University of Groningen"
    persona: str = Persona(name="RugHbLlm", avatar_route="api/ai/static/jupyternaut.svg")

    def __init__(self, **kwargs):
        # Fetch the openai_api_base from the provided configuration or environment variable
        self.openai_api_base = kwargs.get(
            "openai_api_base",
            getenv("RUGHB_API_BASE", "http://localhost:8000/v1"),
        )
        super().__init__(**kwargs)

class RugLiteLlm(BaseProvider, OpenAI):
    id = "ruglitellm"
    name = "RugLiteLlm"
    models = [
        "mistral-nemo-instruct",
        "llama-3.1-8b-instruct-fp8"
    ]
    help = "Click here for more details on [RugLlm](https://rug.nl)"
    model_id_key = "model_name"
    model_id_label = "Model ID"
    pypi_package_deps = ["langchain_openai"]
    auth_strategy = EnvAuthStrategy(
        name="RUGLITELLM_API_KEY", keyword_param="openai_api_key",
    )
    openai_api_base: str = getenv("RUGLITELLM_API_BASE", 'https://llm.hpc.rug.nl/')
    openai_organization: str = "University of Groningen"
    persona: str = Persona(name="RugLiteLlm", avatar_route="api/ai/static/jupyternaut.svg")

class ChatRugLiteLlm(BaseProvider, ChatOpenAI):
    id = "ruglitellm"
    name = "RugLiteLlm"
    models = [
        "mistral-nemo-instruct",
        "llama-3.1-8b-instruct-fp8"
    ]
    help = "Click here for more details on [RugLlm](https://rug.nl)"
    model_id_key = "model_name"
    model_id_label = "Model ID"
    pypi_package_deps = ["langchain_openai"]
    auth_strategy = EnvAuthStrategy(
        name="RUGLITELLM_API_KEY", keyword_param="openai_api_key",
    )
    openai_api_base: str = getenv("RUGLITELLM_API_BASE", 'https://llm.hpc.rug.nl/')
    openai_organization: str = "University of Groningen"
    persona: str = Persona(name="RugLiteLlm", avatar_route="api/ai/static/jupyternaut.svg")



class LlmGlfhProvider(BaseProvider, OpenAI):
    id = "glfh"
    name = "Glfh"
    models = [
        "hf:mistralai/Mistral-7B-Instruct-v0.3",
        "hf:google/gemma-2-27b-it",
        "hf:Qwen/Qwen2.5-72B-Instruct",
        "hf:NousResearch/Hermes-3-Llama-3.1-70B",
        "hf:mlabonne/Llama-3.1-70B-Instruct-lorablated",
        "hf:01-ai/Yi-34B",
        "hf:01-ai/Yi-34B-Chat",
        "hf:Gryphe/MythoMax-L2-13b",
        "hf:NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "hf:NousResearch/Nous-Hermes-2-Yi-34B",
        "hf:NousResearch/Nous-Hermes-Llama2-13b",
        "hf:Qwen/Qwen1.5-110B-Chat",
        "hf:Qwen/Qwen1.5-14B-Chat",
        "hf:Qwen/Qwen1.5-72B-Chat",
        "hf:allenai/OLMo-7B-Twin-2T-hf",
        "hf:allenai/OLMo-7B-hf",
        "hf:alpindale/WizardLM-2-8x22B",
        "hf:databricks/dbrx-instruct",
        "hf:deepseek-ai/deepseek-llm-67b-chat",
        "hf:google/gemma-2-27b-it",
        "hf:google/gemma-2-9b-it",
        "hf:google/gemma-2b-it",
        "hf:meta-llama/Meta-Llama-3-8B-Instruct",
        "hf:meta-llama/Meta-Llama-3.1-405B-Instruct",
        "hf:meta-llama/Meta-Llama-3.1-70B-Instruct",
        "hf:meta-llama/Meta-Llama-3.1-8B-Instruct",
        "hf:mistralai/Mistral-7B-Instruct-v0.3",
        "hf:mistralai/Mixtral-8x22B-Instruct-v0.1",
        "hf:mistralai/Mixtral-8x7B-Instruct-v0.1",
        "hf:togethercomputer/StripedHyena-Nous-7B",
        "hf:upstage/SOLAR-10.7B-Instruct-v1.0",
    ]
    help = "Click here for more details on [glfh](https://glhf.chat)"
    model_id_key = "model_name"
    model_id_label = "Model ID"
    pypi_package_deps = ["langchain_openai"]
    auth_strategy = EnvAuthStrategy(
        name="GLFH_API_KEY", keyword_param="openai_api_key",
    )
    openai_api_base: str = getenv("GLFH_API_BASE", 'https://glhf.chat/api/openai/v1')
    # openai_proxy = getenv("RUGLLM_PROXY","http://web:8010/v1")
  
    openai_organization: str = "GLFH"
    persona: str = Persona(name="Glfh", avatar_route="api/ai/static/jupyternaut.svg")

    @classmethod
    def is_api_key_exc(cls, e: Exception):
        """
        Determine if the exception is an RugLlm API key error.
        """
        import openai
        if isinstance(e, openai.AuthenticationError):
            error_details = e.json_body.get("error", {})
            return error_details.get("code") == "invalid_api_key"
        return False

class ChatLlmGlfhProvider(BaseProvider, ChatOpenAI):
    id = "glfh"
    name = "Glfh"
    models = [
        "hf:mistralai/Mistral-7B-Instruct-v0.3",
        "hf:google/gemma-2-27b-it",
        "hf:Qwen/Qwen2.5-72B-Instruct",
        "hf:NousResearch/Hermes-3-Llama-3.1-70B",
        "hf:mlabonne/Llama-3.1-70B-Instruct-lorablated",
        "hf:01-ai/Yi-34B",
        "hf:01-ai/Yi-34B-Chat",
        "hf:Gryphe/MythoMax-L2-13b",
        "hf:NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "hf:NousResearch/Nous-Hermes-2-Yi-34B",
        "hf:NousResearch/Nous-Hermes-Llama2-13b",
        "hf:Qwen/Qwen1.5-110B-Chat",
        "hf:Qwen/Qwen1.5-14B-Chat",
        "hf:Qwen/Qwen1.5-72B-Chat",
        "hf:allenai/OLMo-7B-Twin-2T-hf",
        "hf:allenai/OLMo-7B-hf",
        "hf:alpindale/WizardLM-2-8x22B",
        "hf:databricks/dbrx-instruct",
        "hf:deepseek-ai/deepseek-llm-67b-chat",
        "hf:google/gemma-2-27b-it",
        "hf:google/gemma-2-9b-it",
        "hf:google/gemma-2b-it",
        "hf:meta-llama/Meta-Llama-3-8B-Instruct",
        "hf:meta-llama/Meta-Llama-3.1-405B-Instruct",
        "hf:meta-llama/Meta-Llama-3.1-70B-Instruct",
        "hf:meta-llama/Meta-Llama-3.1-8B-Instruct",
        "hf:mistralai/Mistral-7B-Instruct-v0.3",
        "hf:mistralai/Mixtral-8x22B-Instruct-v0.1",
        "hf:mistralai/Mixtral-8x7B-Instruct-v0.1",
        "hf:togethercomputer/StripedHyena-Nous-7B",
        "hf:upstage/SOLAR-10.7B-Instruct-v1.0",    ]
    help = "Click here for more details on [glfh](https://glfh.chat)"
    model_id_key = "model_name"
    model_id_label = "Model ID"
    pypi_package_deps = ["langchain_openai"]
    auth_strategy = EnvAuthStrategy(
        name="GLFH_API_KEY", keyword_param="openai_api_key",
    )
    openai_api_base: str = getenv("GLFH_API_BASE", 'https://glhf.chat/api/openai/v1')
    # openai_proxy = getenv("RUGLLM_PROXY","http://web:8010/v1")
    openai_organization: str = "GLFH"
    persona: str = Persona(name="Glfh", avatar_route="api/ai/static/jupyternaut.svg")

    @classmethod
    def is_api_key_exc(cls, e: Exception):
        """
        Determine if the exception is an RugLlm API key error.
        """
        import openai
        if isinstance(e, openai.AuthenticationError):
            error_details = e.json_body.get("error", {})
            return error_details.get("code") == "invalid_api_key"
        return False

class OpenRouterProvider(BaseProvider, OpenAI):
    id = "openrouter"
    name = "OpenRouter"
    models = [
            "mistralai/pixtral-12b:free",
            "qwen/qwen-2-vl-7b-instruct:free",
            "nousresearch/hermes-3-llama-3.1-405b:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "qwen/qwen-2-7b-instruct:free",
            "google/gemma-2-9b-it:free",
            "mistralai/mistral-7b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "microsoft/phi-3-medium-128k-instruct:free",
            "meta-llama/llama-3-8b-instruct:free",
            "gryphe/mythomist-7b:free",
            "openchat/openchat-7b:free",
            "undi95/toppy-m-7b:free",
            "huggingfaceh4/zephyr-7b-beta:free",
    ]
    help = "Click here for more details on [OpenRouter](https://openrouter.ai/)"
    model_id_key = "model_name"
    model_id_label = "Model ID"
    pypi_package_deps = ["langchain_openai", "openai", "requests"]
    auth_strategy = EnvAuthStrategy(
        name="OPENROUTER_API_KEY", keyword_param="open_api_key",
    )
    # openai_api_key = getenv("RUGLLM_API_KEY", 'Empty')
    openai_api_base: str = getenv("OPENROUTER_API_BASE", 'https://openrouter.ai/api/v1')
    openai_organization: str = "Open Router"
    persona: str = Persona(name="OpenRouter", avatar_route="api/ai/static/jupyternaut.svg")

class ChatOpenRouterProvider(BaseProvider, ChatOpenAI):
    id = "openrouter"
    name = "OpenRouter"
    models = [
            "mistralai/pixtral-12b:free",
            "qwen/qwen-2-vl-7b-instruct:free",
            "nousresearch/hermes-3-llama-3.1-405b:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "qwen/qwen-2-7b-instruct:free",
            "google/gemma-2-9b-it:free",
            "mistralai/mistral-7b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "microsoft/phi-3-medium-128k-instruct:free",
            "meta-llama/llama-3-8b-instruct:free",
            "gryphe/mythomist-7b:free",
            "openchat/openchat-7b:free",
            "undi95/toppy-m-7b:free",
            "huggingfaceh4/zephyr-7b-beta:free",
    ]
    help = "Click here for more details on [OpenRouter](https://openrouter.ai/)"
    model_id_key = "model_name"
    model_id_label = "Model ID"
    pypi_package_deps = ["langchain_openai", "openai", "requests"]
    auth_strategy = EnvAuthStrategy(
        name="OPENROUTER_API_KEY", keyword_param="open_api_key",
    )
    openai_api_base: str = getenv("OPENROUTER_API_BASE", 'https://openrouter.ai/api/v1')
    openai_organization: str = "Open Router"
    persona: str = Persona(name="OpenRouter", avatar_route="api/ai/static/jupyternaut.svg")

