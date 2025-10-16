# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import os
from datetime import datetime

class Score(BaseModel):
    name: str
    value: Any

class Prediction(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict)
    output: Any = ""

class EvalPayload(BaseModel):
    eval_name: Optional[str] = None
    group: Optional[str] = None
    prediction: Prediction
    scores: List[Score] = Field(default_factory=list)

app = FastAPI()

def _wandb_bootstrap() -> Dict[str, str]:
    """
    1) Usa WEAVE_WANDB_KEY para iniciar sesión en W&B (crea ~/.netrc).
    2) Elimina WANDB_API_KEY del entorno para evitar conflicto env+netrc.
    3) Devuelve {'project', 'entity'}.
    """
    api_key = os.getenv("WEAVE_WANDB_KEY", "").strip()
    if not api_key:
        raise RuntimeError("WEAVE_WANDB_KEY no configurada")

    # Flags de servicio
    os.environ.setdefault("WANDB_MODE", "online")
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")

    import wandb
    # Crea/actualiza ~/.netrc
    wandb.login(key=api_key, relogin=True)

    project = os.getenv("WEAVE_PROJECT", "n8n-RAG").strip()

    # (OPCIONAL) Inicializar Weave para dashboards, pero NO es necesario para loguear métricas
    # from weave import init as weave_init
    # weave_init(project)

    return {"project": project}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/log-eval")
def log_eval(payload: EvalPayload):
    try:
        cfg = _wandb_bootstrap()

        import wandb

        run_name = payload.eval_name or f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run = wandb.init(
            project=cfg["project"],
            group=payload.group,
            name=run_name,
            job_type="evaluation",
            config=payload.prediction.inputs,  # guarda inputs como config
            reinit=True,
        )

        table = wandb.Table(columns=["inputs", "output"])
        table.add_data(payload.prediction.inputs, payload.prediction.output)
        wandb.log({"prediction": table})

        metrics = {s.name: s.value for s in payload.scores}
        if metrics:
            wandb.log(metrics)

        wandb.summary["n_predictions"] = 1
        run.finish()

        return {
            "status": "ok",
            "project": cfg["project"],
            "eval_name": run_name,
            "group": payload.group,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
