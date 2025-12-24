import os
from fastapi import FastAPI, HTTPException
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import sys
from schemas import TestRunResponse,TestRunDetailsResponse,FilterResponse,AllFiltersResponse
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
    "/test-runs/{run_name}/details",
    response_model=list[TestRunDetailsResponse]
)
def get_test_run_details(run_name: str):
    try:
        db = DB(db_url=db_url, debug=False)

        details = db.get_all_run_details_by_run_name(run_name)

        return [
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
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))