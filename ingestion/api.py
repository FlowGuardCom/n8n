# ingestion/api.py
import os
import asyncio
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RAG Ingestion API")

class IngestRequest(BaseModel):
    documents_path: str = "/app/documents"
    client: Optional[str] = None
    clean: bool = False  # si quieres vaciar tablas antes

async def _run_ingest_module(documents_path: str, clean: bool) -> tuple[int, str, str]:
    """
    Lanza `python -m ingestion.ingest --documents <path> [--clean]`
    y devuelve (returncode, stdout, stderr).
    """
    args = ["python", "-m", "ingestion.ingest", "--documents", documents_path]
    if clean:
        args.append("--clean")

    # Log: útil para depurar
    print(f"[ingestion.api] launching: {' '.join(args)}")

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/app",
        env=os.environ.copy(),
    )
    out_bytes, err_bytes = await proc.communicate()
    return proc.returncode, out_bytes.decode(errors="replace"), err_bytes.decode(errors="replace")

@app.post("/ingest")
async def ingest(req: IngestRequest):
    # Si llega `client`, concatenamos subcarpeta
    target_path = req.documents_path
    if req.client:
        target_path = os.path.join(target_path, req.client)

    if not os.path.exists(target_path):
        return {"status": "error", "message": f"Path not found: {target_path}"}

    rc, out, err = await _run_ingest_module(target_path, req.clean)

    return {
        "status": "ok" if rc == 0 else "error",
        "code": rc,
        "path": target_path,
        "stdout": out[-10000:],  # cap por si hay logs enormes
        "stderr": err[-10000:],
    }
