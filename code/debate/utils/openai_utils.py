try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None


model2max_context = {
    "gpt-4": 7900,
    "gpt-4-0314": 7900,
    "gpt-3.5-turbo-0301": 3900,
    "gpt-3.5-turbo": 3900,
    "text-davinci-003": 4096,
    "text-davinci-002": 4096,

    # ---- Local / Ollama models (best-effort defaults; adjust as needed) ----
    # NOTE: Ollama's effective context is controlled by the model and runtime options.
    # If you see context-related errors, set OLLAMA_NUM_CTX (see agent.py) or lower your
    # conversation length.
    "qwen3:14b": 32768,
    "qwen3:32b": 32768,
    "deepseek-r1:8b": 32768,
    "qwen2.5:7b": 32768,
}

class OutOfQuotaException(Exception):
    "Raised when the key exceeded the current quota"
    def __init__(self, key, cause=None):
        super().__init__(f"No quota for key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

class AccessTerminatedException(Exception):
    "Raised when the key has been terminated"
    def __init__(self, key, cause=None):
        super().__init__(f"Access terminated key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    # tiktoken is optional. If unavailable, or the model name is unknown, fall back to a rough estimate.
    if tiktoken is None:
        return max(1, len(string) // 4)
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        # Use a reasonable default encoding when model_name isn't known to tiktoken.
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

