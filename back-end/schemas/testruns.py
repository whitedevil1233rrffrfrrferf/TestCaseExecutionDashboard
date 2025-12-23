from pydantic import BaseModel
from typing import Optional

class TestRunResponse(BaseModel):
    run_id: int
    run_name: str
    target: str
    status: str
    start_ts: str
    end_ts: Optional[str]
    domain: Optional[str] 