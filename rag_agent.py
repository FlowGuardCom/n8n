"""
RAG CLI simple con Ollama (sin pydantic-ai).
"""

import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# 1) Cargar .env primero
load_dotenv(".env")

# 2) Forzar proveedor Ollama ANTES de importar providers
#    - Establecemos valores por defecto si no existen
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("CHAT_MODEL", os.getenv("CHAT_MODEL", "llama3.1"))
os.environ.setdefault("OLLAMA_URL", os.getenv("OLLAMA_URL", "http://ollama:11434"))

# (opcional pero recomendable) evitar que el selector elija OpenAI
# si tu utils.providers prioriza OPENAI_API_KEY cuando existe:
if os.environ.get("LLM_PROVIDER", "").lower() == "ollama":
    os.environ.pop("OPENAI_API_KEY", None)

from utils.providers import get_chat_backend, validate_configuration
from cli import (
    search_knowledge_base,
    build_rag_messages,
    initialize_db,
    close_db,
)  # reutilizamos

logger = logging.getLogger(__name__)


async def run_cli():
    # get_chat_backend ahora devolverá el backend de Ollama por LLM_PROVIDER=ollama
    chat = get_chat_backend()
    await initialize_db()

    provider = os.environ.get("LLM_PROVIDER", "ollama")
    chat_model = os.environ.get("CHAT_MODEL", "llama3.1")
    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")

    print("=" * 60)
    print("🤖 Docling RAG Knowledge Assistant (Ollama)")
    print("=" * 60)
    print(f"Provider: {provider}  |  Chat model: {chat_model}  |  OLLAMA_URL: {ollama_url}")
    print("Type 'exit' or 'quit' to leave.")
    print("=" * 60)

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break

        hits = await search_knowledge_base(q, limit=5)
        messages = build_rag_messages(q, hits)
        print("Assistant: ", end="", flush=True)
        async for chunk in chat.chat_stream(messages):
            print(chunk, end="", flush=True)
        print()


async def main():
    logging.basicConfig(level=logging.INFO)
    if not validate_configuration():
        # En caso de que validate_configuration exija OPENAI_API_KEY, puedes adaptar allí
        # para que acepte el modo Ollama (recomendado).
        sys.exit(1)
    try:
        await run_cli()
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
