from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
import os

class Settings(BaseModel):
    # data / modeling
    csv_path: str = Field(default=os.getenv("CSV_PATH", "data/dynamic_supply_chain_logistics_dataset.csv"))
    target_col: str = Field(default=os.getenv("TARGET_COL", "risk_classification"))
    random_state: int = Field(default=int(os.getenv("RANDOM_STATE", "42")))
    # external
    openai_api_key: Optional[str] = Field(default=os.getenv("OPENAI_API_KEY"))
    slack_webhook_url: Optional[str] = Field(default=os.getenv("SLACK_WEBHOOK_URL"))
    # server
    host: str = Field(default=os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default=int(os.getenv("PORT", "8050")))
    debug: bool = Field(default=os.getenv("DEBUG", "1") == "1")

settings = Settings()
