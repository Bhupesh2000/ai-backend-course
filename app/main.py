import os, time, json, uuid, logging
from typing import AsyncIterator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(message)s")

MODEL       = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "500"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
TIMEOUT_S   = float(os.getenv("LLM_TIMEOUT_S", "30"))

client = AsyncOpenAI(timeout=TIMEOUT_S)
app = FastAPI(title="AI Backend Course — Week 2 Service")

class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    system: str | None = None

class AnswerResponse(BaseModel):
    request_id: str
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int

def build_messages(req: AnswerRequest):
    msgs = [{"role": "system",
             "content": req.system or "You are a terse senior backend engineer."}]
    # user content is DELIMITED — Day 5, and the foundation of Day 40
    msgs.append({"role": "user",
                 "content": f"<question>{req.question}</question>"})
    return msgs

def emit(**kw):
    log.info(json.dumps(kw))

@app.post("/v1/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest):
    rid = str(uuid.uuid4())
    t0 = time.perf_counter()
    try:
        r = await client.chat.completions.create(
            model=MODEL, messages=build_messages(req),
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
        )
    except Exception as e:
        emit(request_id=rid, route="answer", error=type(e).__name__)
        raise HTTPException(status_code=503,
                            detail={"code": "UPSTREAM_UNAVAILABLE",
                                    "request_id": rid})

    choice = r.choices[0]
    if choice.finish_reason == "length":
        emit(request_id=rid, route="answer", error="TRUNCATED")
        raise HTTPException(status_code=502,
                            detail={"code": "TRUNCATED_OUTPUT",
                                    "request_id": rid})

    ms = int((time.perf_counter() - t0) * 1000)
    emit(request_id=rid, route="answer", model=r.model,
         input_tokens=r.usage.prompt_tokens,
         output_tokens=r.usage.completion_tokens,
         latency_ms=ms, finish_reason=choice.finish_reason)

    return AnswerResponse(
        request_id=rid, answer=choice.message.content, model=r.model,
        input_tokens=r.usage.prompt_tokens,
        output_tokens=r.usage.completion_tokens, latency_ms=ms,
    )

@app.post("/v1/answer/stream")
async def answer_stream(req: AnswerRequest):
    rid = str(uuid.uuid4())

    async def gen() -> AsyncIterator[str]:
        t0 = time.perf_counter()
        ttft_ms = None
        try:
            stream = await client.chat.completions.create(
                model=MODEL, messages=build_messages(req),
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                stream=True, stream_options={"include_usage": True},
            )
            usage = None
            async for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content
                if piece:
                    if ttft_ms is None:
                        ttft_ms = int((time.perf_counter() - t0) * 1000)
                    yield f"data: {json.dumps({'delta': piece})}\n\n"
            emit(request_id=rid, route="stream", ttft_ms=ttft_ms,
                 latency_ms=int((time.perf_counter() - t0) * 1000),
                 output_tokens=usage.completion_tokens if usage else None)
        except Exception as e:
            # HTTP 200 already sent — error must be IN-BAND (Day 11)
            emit(request_id=rid, route="stream", error=type(e).__name__)
            yield f"data: {json.dumps({'error': 'UPSTREAM_UNAVAILABLE'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"X-Request-Id": rid,
                                      "Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})  # nginx

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/readyz")
async def readyz():
    try:
        await client.models.retrieve(MODEL)
        return {"status": "ready", "model": MODEL}
    except Exception:
        raise HTTPException(status_code=503, detail={"code": "LLM_UNAVAILABLE"})