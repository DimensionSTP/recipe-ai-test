import dotenv

dotenv.load_dotenv(
    override=True,
)

from typing import Dict, Any, Optional
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

import hydra
from omegaconf import DictConfig

from src.utils import SetUp


class RecommendIn(BaseModel):
    lab_id: str
    category_value: Optional[str] = None


class RecommendOut(BaseModel):
    result: Any


app = FastAPI(title="Recipe-AI Recommend API")

API_KEY = os.getenv("API_KEY", "")
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "9001"))


def _auth(authorization: str = Header(default="")) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )


_app_state: Dict[str, Any] = {"manager": None}


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/recommend",
    response_model=RecommendOut,
)
def recommend_api(body: RecommendIn, _=Depends(_auth)) -> RecommendOut:
    if _app_state["manager"] is None:
        raise HTTPException(
            status_code=500,
            detail="Manager not initialized.",
        )
    try:
        result = _app_state["manager"].recommend(
            lab_id=body.lab_id,
            category_value=body.category_value,
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@hydra.main(
    config_path="../configs",
    config_name="main.yaml",
)
def main(
    config: DictConfig,
) -> None:
    setup = SetUp(config)
    manager = setup.get_manager(manager_type="recommendation")
    _app_state["manager"] = manager

    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
