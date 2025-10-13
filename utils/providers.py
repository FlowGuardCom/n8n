# utils/providers.py
import os
import httpx
import asyncio


# ----------------------------
# Chat backend (ya lo tenías)
# ----------------------------
class OllamaChat:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def chat_stream(self, messages):
        async with httpx.AsyncClient(timeout=120) as client:
            req = {"model": self.model, "messages": messages, "stream": True}
            async with client.stream("POST", f"{self.base_url}/api/chat", json=req) as r:
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        chunk = line[5:].strip()
                        # La respuesta de Ollama ya viene como JSON por línea
                        try:
                            data = httpx.Response(200, text=chunk).json()
                        except Exception:
                            continue
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]

def get_chat_backend():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        return OllamaChat(
            base_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
            model=os.getenv("CHAT_MODEL", "llama3.1"),
        )
    raise RuntimeError("LLM_PROVIDER debe ser 'ollama'")

def get_model_info():
    return {
        "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
        "llm_model": os.getenv("CHAT_MODEL", "llama3.1"),
        "embedding_model": os.getenv("EMBED_MODEL", "nomic-embed-text"),
    }

def validate_configuration():
    if os.getenv("LLM_PROVIDER", "ollama").lower() != "ollama":
        print("Solo está soportado LLM_PROVIDER=ollama en esta build.")
        return False
    return True


# ----------------------------
# Embeddings (nuevo)
# ----------------------------
class OllamaEmbeddingClient:
    """
    Cliente sencillo para /api/embed de Ollama.
    """
    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        """
        Devuelve una lista de embeddings (uno por entrada).
        """
        if not inputs:
            return []

        payload = {"model": self.model, "input": inputs}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/embed", json=payload)
            resp.raise_for_status()
            data = resp.json()
            # Ollama: {"model": "...", "embeddings": [[...], [...], ...]}
            return data.get("embeddings", [])

    async def embed_one(self, text: str) -> list[float]:
        embs = await self.embed([text])
        return embs[0] if embs else []


def get_embedding_model() -> str:
    """
    Nombre del modelo de embeddings (por defecto: nomic-embed-text).
    """
    return os.getenv("EMBED_MODEL", "nomic-embed-text")


def get_embedding_client():
    """
    Devuelve un cliente de embeddings listo para usar.
    """
    base_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
    model = get_embedding_model()
    return OllamaEmbeddingClient(base_url=base_url, model=model)
