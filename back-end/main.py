import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openpyxl import Workbook
import tempfile
import os
import mysql.connector
import uvicorn
from mysql.connector import Error
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import sys
from schemas import TestRunResponse,TestRunDetailsResponse,FilterResponse,AllFiltersResponse,TestRunSummaryResponse,TestRunFullResponse,RunEvaluationSummaryResponse,EvaluationItemResponse,ConversationResponse,TestCaseResponse,FullConversationResponse, TimelineEvent
load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lib.orm import DB

db_url = (
            f"mysql+mysqlconnector://"
            f"{os.getenv('DB_USER')}:"
            f"{os.getenv('DB_PASSWORD')}@"
            f"{os.getenv('DB_HOST')}:"
            f"{os.getenv('DB_PORT')}/"
            f"{os.getenv('DB_NAME')}"
        )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(
    "/get_all_test_runs",
    response_model=list[TestRunResponse]
)
def get_all_test_runs():
    try:
        db = DB(db_url=db_url, debug=False)
        runs = db.get_all_runs()

        response = []

        for r in runs:
            domain_name = None

            target_id = r.kwargs.get("target_id") if hasattr(r, "kwargs") else None

            if target_id:
                target = db.get_target_by_id(target_id)
                if target:
                    domain_name = target.target_domain   # ✅ FIX

            response.append(
                TestRunResponse(
                    run_id=r.run_id,
                    run_name=r.run_name,
                    target=r.target,
                    status=r.status,
                    start_ts=r.start_ts,
                    end_ts=r.end_ts,
                    domain=domain_name
                )
            )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get(
    "/test-runs/{run_name}",
    response_model=TestRunFullResponse
)
def get_test_run(run_name: str):
    try:
        db = DB(db_url=db_url, debug=False)

        # ---------- RUN SUMMARY ----------
        run = db.get_run_by_name(run_name)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        domain_name = None
        if getattr(run, "target_id", None):
            target = db.get_target_by_id(run.target_id)
            if target:
                domain_name = getattr(target, "target_domain", None)

        summary = TestRunSummaryResponse(
            run_id=run.run_id,
            run_name=run.run_name,
            target=run.target,
            domain=domain_name,
            status=run.status,
            start_ts=run.start_ts,
            end_ts=run.end_ts
        )

        # ---------- RUN DETAILS ----------
        details = db.get_all_run_details_by_run_name(run_name)

        details_response = [
            TestRunDetailsResponse(
                run_name=d.run_name,
                testcase_name=d.testcase_name,
                metric_name=d.metric_name,
                plan_name=d.plan_name,
                conversation_id=d.conversation_id,
                status=d.status,
                detail_id=d.detail_id
            )
            for d in details
        ]

        # ---------- FINAL RESPONSE ----------
        return TestRunFullResponse(
            summary=summary,
            details=details_response
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_all_filters", response_model=AllFiltersResponse)
def get_all_filters():
    try:
        db = DB(db_url=db_url, debug=False)  # your DB instance

        # Use the @property methods to get all data
        return AllFiltersResponse(
            domains=[FilterResponse(filter_name=d.name) for d in db.domains],
            languages=[FilterResponse(filter_name=l.name) for l in db.languages],
            targets=[FilterResponse(filter_name=t.target_name) for t in db.targets],
            plans=[FilterResponse(filter_name=p.plan_name) for p in db.plans],
            metrics=[FilterResponse(filter_name=m.metric_name) for m in db.metrics]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/test-runs/{run_name}/summary", response_model=TestRunSummaryResponse)
def get_test_run_summary(run_name: str):
    try:
        db = DB(db_url=db_url, debug=False)

        run = db.get_run_by_name(run_name)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        # If you store target_id on run, use it to fetch target -> domain
        domain_name = None
        if getattr(run, "target_id", None):
            target = db.get_target_by_id(run.target_id)
            if target:
                domain_name = getattr(target, "target_domain", None)

        return TestRunSummaryResponse(
            run_id=run.run_id,
            run_name=run.run_name,
            target=run.target,
            domain=domain_name,
            status=run.status,
            start_ts=run.start_ts,
            end_ts=run.end_ts
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    
    
@app.get(
    "/test-runs/{run_name}/evaluation-summary",
    response_model=RunEvaluationSummaryResponse
)
def get_run_evaluation_summary(run_name: str):
    try:
        db = DB(db_url=db_url, debug=False)
        run= db.get_run_by_name(run_name)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        domain_name = None
        if getattr(run, "target_id", None):
            target = db.get_target_by_id(run.target_id)
            if target:
                domain_name = getattr(target, "target_domain", None)
        run_summary = TestRunSummaryResponse(
            run_id=run.run_id,
            run_name=run.run_name,
            target=run.target,
            domain=domain_name,
            status=run.status,
            start_ts=run.start_ts,
            end_ts=run.end_ts
        )        
        details = db.get_all_run_details_by_run_name(run_name)

        evaluations=[]

        for d in details:
            conv = db.get_conversation_by_id(d.conversation_id)
            print(conv.evaluation_score)
            if not conv:
                continue
            evaluations.append(
                EvaluationItemResponse(
                    detail_id=d.detail_id,
                    testcase=conv.testcase,
                    agent_response=conv.agent_response,
                    evaluation_score=conv.evaluation_score,
                    evaluation_reason=conv.evaluation_reason,
                    evaluation_ts=conv.evaluation_ts
                )
            )

        return RunEvaluationSummaryResponse(
            run=run_summary,
            evaluations=evaluations
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test-runs/{run_name}/evaluation-report")
def download_evaluation_report(run_name: str):
    try:
        db = DB(db_url=db_url, debug=False)

        # -------- Run summary --------
        run = db.get_run_by_name(run_name)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        domain_name = None
        if getattr(run, "target_id", None):
            target = db.get_target_by_id(run.target_id)
            if target:
                domain_name = getattr(target, "target_domain", None)

        # -------- Get details --------
        details = db.get_all_run_details_by_run_name(run_name)

        # -------- Create Excel --------
        wb = Workbook()
        ws = wb.active
        ws.title = "Evaluation Report"

        # -------- Run summary section --------
        ws.append(["Run Name", run.run_name])
        ws.append(["Target", run.target])
        ws.append(["Domain", domain_name])
        ws.append(["Status", run.status])
        ws.append(["Start Time", run.start_ts])
        ws.append(["End Time", run.end_ts])
        ws.append([])  # empty row

        # -------- Table header --------
        ws.append([
            "Detail ID",
            "Testcase",
            "Agent Response",
            "Evaluation Score",
            "Evaluation Reason",
            "Evaluation Time"
        ])

        # -------- Rows --------
        for d in details:
            conv = db.get_conversation_by_id(d.conversation_id)
            if not conv:
                continue

            ws.append([
                d.detail_id,
                conv.testcase,
                conv.agent_response,
                conv.evaluation_score,
                conv.evaluation_reason,
                conv.evaluation_ts
            ])

        # -------- Save temp file --------
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        wb.save(tmp_file.name)
        tmp_file.close()

        return FileResponse(
            path=tmp_file.name,
            filename=f"{run_name}_evaluation_report.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))     


@app.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse
)
def get_conversation(conversation_id: int):
    db = DB(db_url=db_url, debug=False)
    conversation = db.get_conversation_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation

@app.get(
    "/testcases/{testcase_name}",
    response_model=TestCaseResponse
)
def get_conversation(testcase_name: str):
    db = DB(db_url=db_url, debug=False)
    testcase = db.get_testcase_by_name(testcase_name)
    if not testcase:
        raise HTTPException(status_code=404, detail="Testcase not found")
    return TestCaseResponse(
        user_prompt=testcase.prompt.user_prompt,
        system_prompt=testcase.prompt.system_prompt
    )

@app.get(
    "/conversations/full/{conversation_id}",
    response_model=FullConversationResponse
)
def get_full_conversation(conversation_id: int):
    db = DB(db_url=db_url, debug=False)
    conversation = db.get_conversation_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    testcase_name = conversation.testcase
    testcase = db.get_testcase_by_name(testcase_name)
    if not testcase:
        user_prompt = None
        system_prompt = None
    else:
        user_prompt = getattr(testcase.prompt, "user_prompt", None)
        system_prompt = getattr(testcase.prompt, "system_prompt", None)

    return FullConversationResponse(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        agent_response=conversation.agent_response,
        testcase_name=testcase_name,
        conversation_id=conversation_id,
        target=conversation.target
    )


@app.get("/conversations/{conversation_id}/timeline")
def get_conversation_timeline_api(conversation_id: int):
    db = DB(db_url=db_url, debug=False)
    timeline = db.get_conversation_timeline(conversation_id)

    if not timeline:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return timeline

@app.get(
    "/test-runs/{run_name}/timeline",
    response_model=list[TimelineEvent]
)
def get_test_run_timeline(run_name: str):
    db = DB(db_url=db_url, debug=False)

    timeline = db.get_run_timeline(run_name)
    if not timeline:
        raise HTTPException(status_code=404, detail="No timeline found")

    return timeline


if __name__ == "__main__":
    

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )