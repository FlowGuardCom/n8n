"""
Document embedding generation for vector search.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

import httpx

from openai import RateLimitError, APIError
from dotenv import load_dotenv

from .chunker import DocumentChunk

# Import flexible providers
try:
    from utils.providers import get_embedding_client, get_embedding_model
except ModuleNotFoundError:
    import os, sys
    ROOT = os.path.dirname(os.path.abspath(__file__))  # /app/ingestion
    PROJECT_ROOT = os.path.dirname(ROOT)               # /app
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from utils.providers import get_embedding_client, get_embedding_model
    
# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize client with flexible provider
#embedding_client = get_embedding_client()
#EMBEDDING_MODEL = get_embedding_model()

# ===== Configuración Ollama =====
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
# Valor por defecto razonable para modelos tipo nomic-embed-text
DEFAULT_EMBED_DIM = 768
# Timeout de red
HTTP_TIMEOUT = float(os.getenv("EMBED_HTTP_TIMEOUT", "60"))

class EmbeddingGenerator:
    """Generates embeddings for document chunks."""

    def __init__(
            self,
            model: str = EMBEDDING_MODEL,
            batch_size: int = 100,
            max_retries: int = 3,
            retry_delay: float = 1.0
    ):
        """
        Initialize embedding generator.

        Args:
            model: OpenAI embedding model to use
            batch_size: Number of texts to process in parallel
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Model-specific configurations
        self.model_configs = {
            # OpenAI (por si un día vuelves a usarlo)
            "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
            "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
            "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191},

            # Ollama embeddings más comunes
            "nomic-embed-text": {"dimensions": 768, "max_tokens": 8192},
            "all-minilm": {"dimensions": 384, "max_tokens": 8192},
            "mxbai-embed-large": {"dimensions": 1024, "max_tokens": 8192},
            "bge-base": {"dimensions": 768, "max_tokens": 8192},
            "bge-small": {"dimensions": 384, "max_tokens": 8192},
            "snowflake-arctic-embed": {"dimensions": 1024, "max_tokens": 8192},
        }
        # Dimensión se infiere en runtime tras el primer embedding
        self._inferred_dim: Optional[int] = None

        if model not in self.model_configs:
            logger.warning(f"Unknown model {model}, using default config")
            self.config = {"dimensions": 1536, "max_tokens": 8191}
        else:
            self.config = self.model_configs[model]

        # ---------- utilidades internas ----------

    def _truncate(self, text: str) -> str:
        # Ollama no usa "max_tokens" fijo para embeddings como OpenAI; mantenemos
        # una truncación defensiva para evitar textos descomunales (> 8192 chars).
        if not text:
            return ""
        return text[:8192]

    def _embedding_dim(self) -> int:
        return self._inferred_dim or DEFAULT_EMBED_DIM

    async def _post_embed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/embed", json=payload)
            resp.raise_for_status()
            return resp.json()

    def _normalize_batch_response(self, data: Dict[str, Any]) -> List[List[float]]:
        """
        Soporta tanto respuestas de Ollama:
        - {"embedding": [...]} para input string (uno)
        - {"embeddings": [[...], [...]]} para input array (batch)
        """
        if "embeddings" in data and isinstance(data["embeddings"], list):
            return data["embeddings"]
        if "embedding" in data and isinstance(data["embedding"], list):
            return [data["embedding"]]
        raise ValueError("Respuesta de Ollama no contiene 'embedding' ni 'embeddings'")

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        text = self._truncate(text or "")

        # Reintentos con backoff
        for attempt in range(self.max_retries):
            try:
                payload = {"model": self.model, "input": text}
                data = await self._post_embed(payload)
                vecs = self._normalize_batch_response(data)
                emb = vecs[0]

                # Infiriendo dimensión si no se conoce aún
                if self._inferred_dim is None:
                    self._inferred_dim = len(emb) if isinstance(emb, list) else DEFAULT_EMBED_DIM

                return emb

            except (httpx.HTTPError, ValueError) as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Ollama embed error (final): {e}")
                    # Fallback: vector cero
                    return [0.0] * self._embedding_dim()
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Ollama embed error, retry in {delay}s: {e}")
                await asyncio.sleep(delay)

        '''# Truncate text if too long
        if len(text) > self.config["max_tokens"] * 4:  # Rough token estimation
            text = text[:self.config["max_tokens"] * 4]

        for attempt in range(self.max_retries):
            try:
                response = await embedding_client.embeddings.create(
                    model=self.model,
                    input=text
                )

                return response.data[0].embedding

            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise

                # Exponential backoff for rate limits
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, retrying in {delay}s")
                await asyncio.sleep(delay)

            except APIError as e:
                logger.error(f"OpenAI API error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)

            except Exception as e:
                logger.error(f"Unexpected error generating embedding: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)
                '''

    async def generate_embeddings_batch(
            self,
            texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Filter and truncate texts
        processed_texts = []
        if not texts:
            return []

            # Preprocesado/truncado
        proc = [self._truncate(t or "") for t in texts]
        # Reintentos con backoff para batch
        for attempt in range(self.max_retries):
            try:
                # Intento en lote (si la versión de Ollama lo soporta)
                payload = {"model": self.model, "input": proc}
                data = await self._post_embed(payload)
                vecs = self._normalize_batch_response(data)

                # Inferir dimensión si procede
                if self._inferred_dim is None and vecs and isinstance(vecs[0], list):
                    self._inferred_dim = len(vecs[0])

                # Sanear longitudes
                out = []
                dim = self._embedding_dim()
                for v in vecs:
                    if not isinstance(v, list):
                        out.append([0.0] * dim)
                    elif len(v) != dim:
                        # Normalizar tamaño si algo raro ocurre
                        if len(v) > dim:
                            out.append(v[:dim])
                        else:
                            out.append(v + [0.0] * (dim - len(v)))
                    else:
                        out.append(v)
                # Si el backend devolvió menos/más, ajustamos a la entrada
                if len(out) != len(proc):
                    # Fallback seguro a procesamiento individual
                    raise ValueError("Cuenta de embeddings no coincide con la cuenta de entradas")
                return out

            except (httpx.HTTPError, ValueError) as e:
                if attempt == self.max_retries - 1:
                    logger.warning(f"Fallo en batch Ollama, usando modo individual: {e}")
                    return await self._process_individually(proc)
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Error en batch Ollama, retry in {delay}s: {e}")
                await asyncio.sleep(delay)
        '''
        for text in texts:
            if not text or not text.strip():
                processed_texts.append("")
                continue

            # Truncate if too long
            if len(text) > self.config["max_tokens"] * 4:
                text = text[:self.config["max_tokens"] * 4]

            processed_texts.append(text)

        for attempt in range(self.max_retries):
            try:
                response = await embedding_client.embeddings.create(
                    model=self.model,
                    input=processed_texts
                )

                return [data.embedding for data in response.data]

            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise

                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, retrying batch in {delay}s")
                await asyncio.sleep(delay)

            except APIError as e:
                logger.error(f"OpenAI API error in batch: {e}")
                if attempt == self.max_retries - 1:
                    # Fallback to individual processing
                    return await self._process_individually(processed_texts)
                await asyncio.sleep(self.retry_delay)

            except Exception as e:
                logger.error(f"Unexpected error in batch embedding: {e}")
                if attempt == self.max_retries - 1:
                    return await self._process_individually(processed_texts)
                await asyncio.sleep(self.retry_delay)
                '''

    async def _process_individually(
            self,
            texts: List[str]
    ) -> List[List[float]]:
        """
        Process texts individually as fallback.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        results: List[List[float]] = []
        for t in texts:
            emb = await self.generate_embedding(t)
            results.append(emb)
            # micro-pauses para no saturar
            await asyncio.sleep(0.02)
        '''
        embeddings = []
        for text in texts:
            try:
                if not text or not text.strip():
                    embeddings.append([0.0] * self.config["dimensions"])
                    continue

                embedding = await self.generate_embedding(text)
                embeddings.append(embedding)

                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to embed text: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * self.config["dimensions"])

        return embeddings
        '''
        return results

    async def embed_chunks(
            self,
            chunks: List[DocumentChunk],
            progress_callback: Optional[callable] = None
    ) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of document chunks
            progress_callback: Optional callback for progress updates

        Returns:
            Chunks with embeddings added
        """
        if not chunks:
            return chunks

        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Process chunks in batches
        embedded_chunks = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]

            try:
                # Generate embeddings for this batch
                embeddings = await self.generate_embeddings_batch(batch_texts)

                # Add embeddings to chunks
                for chunk, embedding in zip(batch_chunks, embeddings):
                    # Create a new chunk with embedding
                    embedded_chunk = DocumentChunk(
                        content=chunk.content,
                        index=chunk.index,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        metadata={
                            **chunk.metadata,
                            "embedding_model": self.model,
                            "embedding_generated_at": datetime.now().isoformat()
                        },
                        token_count=chunk.token_count
                    )

                    # Add embedding as a separate attribute
                    embedded_chunk.embedding = embedding
                    embedded_chunks.append(embedded_chunk)

                # Progress update
                current_batch = (i // self.batch_size) + 1
                if progress_callback:
                    progress_callback(current_batch, total_batches)

                logger.info(f"Processed batch {current_batch}/{total_batches}")

            except Exception as e:
                logger.error(f"Failed to process batch {i // self.batch_size + 1}: {e}")

                # Add chunks without embeddings as fallback
                for chunk in batch_chunks:
                    chunk.metadata.update({
                        "embedding_error": str(e),
                        "embedding_generated_at": datetime.now().isoformat()
                    })
                    chunk.embedding = [0.0] * self.config["dimensions"]
                    embedded_chunks.append(chunk)

        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.

        Args:
            query: Search query

        Returns:
            Query embedding
        """
        return await self.generate_embedding(query)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        return self.config["dimensions"]


# Cache for embeddings
class EmbeddingCache:
    """Simple in-memory cache for embeddings."""

    def __init__(self, max_size: int = 1000):
        """Initialize cache."""
        self.cache: Dict[str, List[float]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.max_size = max_size

    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        text_hash = self._hash_text(text)
        if text_hash in self.cache:
            self.access_times[text_hash] = datetime.now()
            return self.cache[text_hash]
        return None

    def put(self, text: str, embedding: List[float]):
        """Store embedding in cache."""
        text_hash = self._hash_text(text)

        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[text_hash] = embedding
        self.access_times[text_hash] = datetime.now()

    def _hash_text(self, text: str) -> str:
        """Generate hash for text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()


# Factory function
def create_embedder(
        model: str = EMBEDDING_MODEL,
        use_cache: bool = True,
        **kwargs
) -> EmbeddingGenerator:
    """
    Create embedding generator with optional caching.

    Args:
        model: Embedding model to use
        use_cache: Whether to use caching
        **kwargs: Additional arguments for EmbeddingGenerator

    Returns:
        EmbeddingGenerator instance
    """
    embedder = EmbeddingGenerator(model=model, **kwargs)

    if use_cache:
        # Add caching capability
        cache = EmbeddingCache()
        original_generate = embedder.generate_embedding

        async def cached_generate(text: str) -> List[float]:
            cached = cache.get(text)
            if cached is not None:
                return cached

            embedding = await original_generate(text)
            cache.put(text, embedding)
            return embedding

        embedder.generate_embedding = cached_generate

    return embedder


# Example usage
async def main():
    """Example usage of the embedder."""
    from .chunker import ChunkingConfig, create_chunker

    # Create chunker and embedder
    config = ChunkingConfig(chunk_size=200, use_semantic_splitting=False)
    chunker = create_chunker(config)
    embedder = create_embedder()

    sample_text =  """
    Docling + Ollama + PGVector pipeline demo for local embeddings without OpenAI usage.
    """

    # Chunk the document
    chunks = chunker.chunk_document(
        content=sample_text,
        title="AI Initiatives",
        source="example.md"
    )

    print(f"Created {len(chunks)} chunks")

    # Generate embeddings
    def progress_callback(current, total):
        print(f"Processing batch {current}/{total}")

    embedded_chunks = await embedder.embed_chunks(chunks, progress_callback)

    for i, chunk in enumerate(embedded_chunks):
        print(f"Chunk {i}: {len(chunk.content)} chars, embedding dim: {len(chunk.embedding)}")

    # Test query embedding
    query_embedding = await embedder.embed_query("Google AI research")
    print(f"Query embedding dimension: {len(query_embedding)}")


if __name__ == "__main__":
    asyncio.run(main())