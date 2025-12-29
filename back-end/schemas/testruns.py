from pydantic import BaseModel
from typing import Optional,List

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

class TestRunSummaryResponse(BaseModel):
    run_id: int
    run_name: str
    target: Optional[str] = None
    domain: Optional[str] = None
    status: str
    start_ts: str
    end_ts: Optional[str] = None


class TestRunFullResponse(BaseModel):
    summary: TestRunSummaryResponse
    details: List[TestRunDetailsResponse]  

class EvaluationItemResponse(BaseModel):
    detail_id: int
    testcase: str
    agent_response: Optional[str]
    evaluation_score: Optional[int]
    evaluation_reason: Optional[str]
    evaluation_ts: Optional[str]


class RunEvaluationSummaryResponse(BaseModel):
    run: TestRunSummaryResponse
    evaluations: List[EvaluationItemResponse]
