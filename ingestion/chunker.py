"""
Docling HybridChunker implementation for intelligent document splitting.

- Carga perezosa de transformers (sólo si vamos a usar Hybrid).
- Fallback simple por párrafos/ventana si no hay Docling/transformers.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# Cargar .env sin sobreescribir variables ya definidas por Docker
load_dotenv(override=False)

logger = logging.getLogger(__name__)

# ---------------- Tokenización ligera (tiktoken opcional) --------------------

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
    _TOKENIZER_NAME = "tiktoken/cl100k_base"

    def _encode(text: str) -> List[int]:
        return _ENC.encode(text or "")

except Exception:
    _ENC = None
    _TOKENIZER_NAME = "heuristic"

    def _encode(text: str) -> List[int]:
        # Heurística aproximada: ~4 chars/token
        text = (text or "").strip()
        approx = max(1, len(text) // 4)
        return list(range(approx))


def count_tokens(text: str) -> int:
    return len(_encode(text))


class _TokenizerAdapter:
    """
    Adaptador con encode() para contar tokens cuando no usamos HF.
    """
    name = _TOKENIZER_NAME

    def encode(self, text: str) -> List[int]:
        return _encode(text or "")


# ---------------- Docling opcional ------------------------------------------

try:
    from docling.chunking import HybridChunker as _DLHybridChunker
    from docling_core.types.doc import DoclingDocument
    _HAS_DOCLING = True
except Exception:
    DoclingDocument = object  # type: ignore
    _DLHybridChunker = None
    _HAS_DOCLING = False


# ---------------- Config y modelos ------------------------------------------

@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    min_chunk_size: int = 100

    use_semantic_splitting: bool = True
    preserve_structure: bool = True

    max_tokens: int = 512

    def __post_init__(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")


@dataclass
class DocumentChunk:
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if self.token_count is None:
            self.token_count = count_tokens(self.content)


# ---------------- Chunkers ---------------------------------------------------

class DoclingHybridChunker:
    """
    Wrapper de Docling HybridChunker + tokenizer HF real para evitar errores de Pydantic.
    Si no se puede cargar el tokenizer, cae a fallback simple.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self._hybrid: Optional[_DLHybridChunker] = None
        self._tokenizer_name = "unknown"

        if not _HAS_DOCLING:
            logger.warning("Docling no está disponible; se usará fallback simple.")
            return

        # Carga perezosa de transformers sólo aquí.
        try:
            from transformers import AutoTokenizer  # type: ignore
            # Modelo ligero con fast tokenizer (cumple BaseTokenizer esperado por Pydantic)
            hf_tok = AutoTokenizer.from_pretrained(
                "bert-base-uncased",
                use_fast=True,
                # si no hay cache previa, intentará descargar; captura excepción más abajo
            )
            self._tokenizer_name = getattr(hf_tok, "name_or_path", "hf_tokenizer")

            # Instancia real del HybridChunker de Docling
            self._hybrid = _DLHybridChunker(
                tokenizer=hf_tok,                # <-- evita el error de Pydantic
                max_tokens=self.config.max_tokens,
                merge_peers=True,
            )
            logger.info(
                f"HybridChunker inicializado "
                f"(max_tokens={self.config.max_tokens}, tokenizer={self._tokenizer_name})"
            )
        except Exception as e:
            logger.error(f"No se pudo inicializar Docling HybridChunker: {e}. Usando fallback simple.")
            self._hybrid = None

        # Tokenizer para contabilizar si hiciera falta en fallback
        self._fallback_tokenizer = _TokenizerAdapter()

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        docling_doc: Optional[DoclingDocument] = None
    ) -> List[DocumentChunk]:
        if not content or not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "hybrid" if (self._hybrid and docling_doc) else "simple_fallback",
            **(metadata or {}),
        }

        if not (self._hybrid and docling_doc):
            if self._hybrid and docling_doc is None:
                logger.warning("No se proporcionó DoclingDocument; usando fallback simple.")
            return self._simple_fallback_chunk(content, base_metadata)

        try:
            chunk_iter = self._hybrid.chunk(dl_doc=docling_doc)
            dl_chunks = list(chunk_iter)

            document_chunks: List[DocumentChunk] = []
            current_pos = 0

            # contextualize() añade jerarquía de encabezados
            for i, ch in enumerate(dl_chunks):
                contextualized_text = self._hybrid.contextualize(chunk=ch)
                tok_count = count_tokens(contextualized_text)

                chunk_meta = {
                    **base_metadata,
                    "total_chunks": len(dl_chunks),
                    "token_count": tok_count,
                    "has_context": True,
                    "tokenizer": self._tokenizer_name,
                    "max_tokens": self.config.max_tokens,
                }

                start_char = current_pos
                end_char = start_char + len(contextualized_text)

                document_chunks.append(
                    DocumentChunk(
                        content=contextualized_text.strip(),
                        index=i,
                        start_char=start_char,
                        end_char=end_char,
                        metadata=chunk_meta,
                        token_count=tok_count,
                    )
                )
                current_pos = end_char

            logger.info(f"Creado {len(document_chunks)} chunks con Docling HybridChunker")
            return document_chunks

        except Exception as e:
            logger.error(f"Fallo en HybridChunker: {e}. Usando fallback simple.")
            return self._simple_fallback_chunk(content, base_metadata)

    def _simple_fallback_chunk(self, content: str, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Ventana deslizante por caracteres, ajustando suavemente a fronteras.
        Si hay tiktoken, limita por tokens (~4 chars/token como guía).
        """
        chunks: List[DocumentChunk] = []
        approx_chars_per_token = 4
        max_chars = self.config.max_tokens * approx_chars_per_token if _ENC else self.config.chunk_size
        overlap_chars = min(self.config.chunk_overlap, max(0, max_chars // 5))

        start = 0
        idx = 0
        n = len(content)

        while start < n:
            end = min(n, start + max_chars)

            if end < n:
                cut = end
                for i in range(end, max(start + self.config.min_chunk_size, end - 200), -1):
                    if i < n and content[i - 1] in ".!?\n":
                        cut = i
                        break
                end = cut

            text = content[start:end].strip()
            if text:
                tok_count = count_tokens(text)
                chunks.append(
                    DocumentChunk(
                        content=text,
                        index=idx,
                        start_char=start,
                        end_char=end,
                        metadata={
                            **base_metadata,
                            "chunk_method": "simple_fallback",
                            "tokenizer": _TOKENIZER_NAME,
                            "max_tokens": self.config.max_tokens,
                            "total_chunks": -1,  # se corrige luego
                            "token_count": tok_count,
                        },
                        token_count=tok_count,
                    )
                )
                idx += 1

            start = end - overlap_chars
            if start <= 0:
                start = end
            if start >= n:
                break

        total = len(chunks)
        for ch in chunks:
            ch.metadata["total_chunks"] = total

        logger.info(f"Fallback simple: creados {total} chunks")
        return chunks


class SimpleChunker:
    """Chunker sencillo por párrafos."""
    def __init__(self, config: ChunkingConfig):
        self.config = config

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[DocumentChunk]:
        if not content or not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "simple",
            "tokenizer": _TOKENIZER_NAME,
            **(metadata or {}),
        }

        import re
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]

        chunks: List[DocumentChunk] = []
        current = ""
        start_pos = 0
        idx = 0

        max_chars = self.config.chunk_size

        for p in paragraphs:
            candidate = f"{current}\n\n{p}" if current else p
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    tok = count_tokens(current)
                    end_pos = start_pos + len(current)
                    chunks.append(
                        DocumentChunk(
                            content=current.strip(),
                            index=idx,
                            start_char=start_pos,
                            end_char=end_pos,
                            metadata={**base_metadata, "token_count": tok},
                            token_count=tok,
                        )
                    )
                    start_pos = end_pos
                    idx += 1
                current = p

        if current:
            tok = count_tokens(current)
            end_pos = start_pos + len(current)
            chunks.append(
                DocumentChunk(
                    content=current.strip(),
                    index=idx,
                    start_char=start_pos,
                    end_char=end_pos,
                    metadata={**base_metadata, "token_count": tok},
                    token_count=tok,
                )
            )

        total = len(chunks)
        for ch in chunks:
            ch.metadata["total_chunks"] = total

        return chunks


# ---------------- Fábrica -----------------------------------------------------

def create_chunker(config: ChunkingConfig):
    """
    Devuelve el chunker apropiado:
      - DoclingHybridChunker si: Docling disponible + transformers disponible + USE_HYBRID_CHUNKER != 0 + config.use_semantic_splitting
      - SimpleChunker en cualquier otro caso.
    """
    use_hybrid_env = os.getenv("USE_HYBRID_CHUNKER", "1").lower() not in ("0", "false", "no")

    if not (use_hybrid_env and config.use_semantic_splitting and _HAS_DOCLING):
        return SimpleChunker(config)

    # Ver si transformers está disponible
    try:
        import transformers  # noqa: F401
        has_hf = True
    except Exception:
        has_hf = False

    if not has_hf:
        logger.warning("transformers no disponible; usando SimpleChunker.")
        return SimpleChunker(config)

    return DoclingHybridChunker(config)
