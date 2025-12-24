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

class TestRunDetailsResponse(BaseModel):
    run_name: str
    testcase_name: str
    metric_name: str
    plan_name: str
    conversation_id: int
    status: str
    detail_id: int    

