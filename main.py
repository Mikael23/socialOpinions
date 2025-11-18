import datetime

from fastapi import FastAPI,Request
from pydantic import BaseModel

from BD.bd_utils import BD
from Utils.preprocessor import TextProcessing
from project_requests.SurveyResponseIn import SurveyResponseIn

app = FastAPI()
textProc  = TextProcessing()
bd = BD()


@app.get("/api/get-survey/{id}")
async def get_survey(id: int):

 return bd.searchBySubject(id)

@app.post("/api/v1/survey-response")
async def receive_survey_response(request: Request):
    raw = await request.body()
    print("RAW BODY:", raw)

    # 2) Try to parse JSON
    try:
        payload = await request.json()
    except Exception as e:
        print("JSON PARSE ERROR:", e)
        return {"status": "error", "message": "Invalid JSON", "raw": raw.decode("utf-8", "ignore")}

    print("PARSED PAYLOAD:", payload)

    text = textProc.preprocess(payload.get('comment'))
    text_after_chunk = textProc.chunk_text(text)
    bd.upsert_chunks(text_after_chunk,payload)

    print("Received:", payload)
    return {"status": "ok", "received": payload}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# def runIngestion(surveys: List[Dict], proc: TextProcessing, store: BD):
#     for survey in surveys:
#         preprocessed = proc.preprocess(survey["text"])
#         chunks = proc.chunk_text(preprocessed)
#         baseMeta = {k: v for k, v in survey.items() if k != "text"}
#         store.upsert_chunks(chunks, baseMeta)

# uvicorn main:app --reload --port 8000

# preprocessed = proc.preprocess(survey["text"])
# chunks = proc.chunk_text(preprocessed)
# baseMeta = {k: v for k, v in survey.items() if k != "text"}
# store.upsert_chunks(chunks, baseMeta)