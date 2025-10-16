#!/usr/bin/env python3
"""
Command Line Interface for Docling RAG Agent (Ollama).

CLI con colores y streaming, sin pydantic-ai.
"""

import asyncio
import asyncpg
import argparse
import logging
import os
import sys
from typing import List, Dict, Any

from dotenv import load_dotenv

from utils.providers import get_chat_backend, validate_configuration, get_model_info
from ingestion.embedder import create_embedder

# Cargar variables
load_dotenv(".env")

logger = logging.getLogger(__name__)

# ANSI color codes
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Pool global de DB
db_pool: asyncpg.pool.Pool | None = None


async def initialize_db():
    """Inicializa el pool de conexiones a PostgreSQL."""
    global db_pool
    if not db_pool:
        db_pool = await asyncpg.create_pool(
            os.getenv("DATABASE_URL"),
            min_size=2,
            max_size=10,
            command_timeout=60
        )


async def close_db():
    """Cierra el pool de conexiones a PostgreSQL."""
    global db_pool
    if db_pool:
        await db_pool.close()
        db_pool = None


async def search_knowledge_base(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Busca en la base de conocimiento por similitud semántica.

    Args:
        query: texto de consulta
        limit: número máximo de resultados

    Returns:
        Lista de dicts con content, document_title, document_source, similarity
    """
    # Asegurar DB
    if not db_pool:
        await initialize_db()

    # Embedding de la query (Ollama nomic-embed-text por create_embedder)
    embedder = create_embedder()
    query_embedding = await embedder.embed_query(query)  # -> list[float]
    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM match_chunks($1::vector, $2)",
            embedding_str,
            limit
        )

    results: list[dict[str, Any]] = []
    for r in rows:
        results.append({
            "content": r["content"],
            "document_title": r["document_title"],
            "document_source": r["document_source"],
            "similarity": float(r["similarity"]),
        })
    return results


def build_rag_messages(user_query: str, hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Construye mensajes estilo OpenAI/Ollama para chat, con contexto RAG.
    """
    system_prompt = (
        "Eres un asistente RAG. Usa SOLO la información de las fuentes proporcionadas.\n"
        "Si no encuentras algo en las fuentes, dilo claramente. Resume y cita las fuentes.\n"
        "Formato de cita: [Source: Título]."
    )

    context_parts: List[str] = []
    for h in hits or []:
        title = h.get("document_title") or "Unknown"
        content = h.get("content") or ""
        context_parts.append(f"[Source: {title}]\n{content}")

    context_block = "\n\n---\n".join(context_parts) if context_parts else "No hay documentos relevantes."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Consulta: {user_query}\n\nContexto:\n{context_block}"}
    ]


class RAGAgentCLI:
    def __init__(self):
        self.chat = get_chat_backend()  # Debe devolver el backend Ollama con .chat_stream()

    def banner(self):
        info = get_model_info()  # Debe leer LLM_PROVIDER/CHAT_MODEL/EMBED_MODEL del entorno
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 60}")
        print("🤖 Docling RAG Knowledge Assistant (Ollama)")
        print("=" * 60)
        print(
            f"{Colors.WHITE}Model: {info.get('llm_model','?')}  |  "
            f"Embeddings: {info.get('embedding_model','?')}  |  "
            f"Provider: {info.get('llm_provider','?')}{Colors.END}"
        )
        print("Type 'exit' or 'quit' to leave.")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.END}\n")

    async def check_db(self) -> bool:
        try:
            await initialize_db()
            async with db_pool.acquire() as conn:
                ok = await conn.fetchval("SELECT 1")
            if ok == 1:
                docs = await self.count("documents")
                chunks = await self.count("chunks")
                print(f"{Colors.GREEN}✓ DB OK{Colors.END} | {docs} documents, {chunks} chunks")
                return True
        except Exception as e:
            print(f"{Colors.RED}✗ DB error: {e}{Colors.END}")
        return False

    async def count(self, table: str) -> int:
        async with db_pool.acquire() as conn:
            return await conn.fetchval(f"SELECT COUNT(*) FROM {table}")

    async def turn(self, user_input: str):
        # 1) Buscar contexto
        hits = await search_knowledge_base(user_input, limit=5)
        # 2) Construir mensajes
        messages = build_rag_messages(user_input, hits)
        # 3) Streaming (Ollama)
        print(f"{Colors.BOLD}Assistant:{Colors.END} ", end="", flush=True)
        async for chunk in self.chat.chat_stream(messages):
            print(chunk, end="", flush=True)
        print()

    async def run(self):
        self.banner()
        if not await self.check_db():
            print(f"{Colors.RED}No DB. Configura DATABASE_URL.{Colors.END}")
            return

        while True:
            try:
                user = input(f"{Colors.BOLD}You:{Colors.END} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{Colors.CYAN}👋 Goodbye!{Colors.END}")
                break

            if not user:
                continue
            if user.lower() in ("exit", "quit", "bye"):
                print(f"{Colors.CYAN}👋 Goodbye!{Colors.END}")
                break

            try:
                await self.turn(user)
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.END}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not validate_configuration():
        sys.exit(1)

    cli = RAGAgentCLI()
    try:
        asyncio.run(cli.run())
    finally:
        try:
            asyncio.run(close_db())
        except RuntimeError:
            # Si el loop ya está cerrado
            pass


if __name__ == "__main__":
    main()
