import dotenv

dotenv.load_dotenv(
    override=True,
)

from typing import Dict, Any
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

import hydra
from omegaconf import DictConfig

from src.utils import SetUp


class ReportIn(BaseModel):
    recommendations: str


class ReportOut(BaseModel):
    text: str


app = FastAPI(title="Recipe-AI Report API")

API_KEY = os.getenv("API_KEY", "")


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
    "/report",
    response_model=ReportOut,
)
def report_api(body: ReportIn, _=Depends(_auth)) -> ReportOut:
    if _app_state["manager"] is None:
        raise HTTPException(
            status_code=500,
            detail="Manager not initialized.",
        )
    try:
        if hasattr(_app_state["manager"], "report"):
            text = _app_state["manager"].report(recommendations=body.recommendations)
        else:
            if (
                not hasattr(_app_state["manager"], "generator")
                or _app_state["manager"].generator is None
            ):
                raise RuntimeError("No report/generator available on manager.")
            text = _app_state["manager"].generator(recommendations=body.recommendations)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@hydra.main(
    config_path="../configs",
    config_name="main.yaml",
)
def main(
    config: DictConfig,
) -> None:
    setup = SetUp(config)
    manager = setup.get_manager(manager_type="report")
    _app_state["manager"] = manager

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.report_port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
