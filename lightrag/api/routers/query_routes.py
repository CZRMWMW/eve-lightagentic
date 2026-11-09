"""
This module contains all query-related routes for the LightRAG API.
"""

import asyncio
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from lightrag.base import QueryParam
from lightrag.api.input_limits import count_conversation_input_chars
from lightrag.api.utils_api import get_combined_auth_dependency, internal_server_error
from lightrag.constants import (
    MAX_KEYWORD_CHARS,
    MAX_KEYWORDS_PER_LIST,
    MAX_MESSAGE_CHARS,
    MAX_MESSAGES_PER_REQUEST,
    MAX_QUERY_CHARS,
    MAX_QUERY_TOKEN_BUDGET,
    MAX_QUERY_TOP_K,
    MAX_REQUEST_TEXT_CHARS,
    MAX_RESPONSE_TYPE_CHARS,
    MAX_ROLE_CHARS,
)
from lightrag.query_validation import validate_query_not_empty, validate_rag_query
from lightrag.utils import logger
from lightrag.api.eve_fittings import (
    dataset_names,
    find_all_fittings,
    query_static_data,
    simulate_fitting,
    list_authorized_characters,
    get_character_skill_data,
    plan_skills,
    query_neo4j,
    query_milvus,
    run_eve_db_call,
    SKILL_NAMES,
    SKILL_DESCRIPTIONS,
    EveQueryAgent,
)
from pydantic import BaseModel, Field, field_validator, model_validator


class QueryRequest(BaseModel):
    query: str = Field(
        max_length=MAX_QUERY_CHARS,
        description=(
            "The query text. Must not be empty. RAG modes additionally require "
            "an English-equivalent length of at least 3; each Chinese, "
            "Japanese or Korean character counts as 2."
        ),
    )

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(
        default="mix",
        description="Query mode",
    )

    only_need_context: Optional[bool] = Field(
        default=None,
        description="If True, only returns the retrieved context without generating a response.",
    )

    only_need_prompt: Optional[bool] = Field(
        default=None,
        description="If True, only returns the generated prompt without producing a response.",
    )

    response_type: Optional[str] = Field(
        min_length=1,
        max_length=MAX_RESPONSE_TYPE_CHARS,
        default=None,
        description="Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'.",
    )

    top_k: Optional[int] = Field(
        ge=1,
        le=MAX_QUERY_TOP_K,
        default=None,
        description="Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode.",
    )

    chunk_top_k: Optional[int] = Field(
        ge=1,
        le=MAX_QUERY_TOP_K,
        default=None,
        description="Number of text chunks to retrieve initially from vector search and keep after reranking.",
    )

    max_entity_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens allocated for entity context in unified token control system.",
        ge=1,
        le=MAX_QUERY_TOKEN_BUDGET,
    )

    max_relation_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens allocated for relationship context in unified token control system.",
        ge=1,
        le=MAX_QUERY_TOKEN_BUDGET,
    )

    max_total_tokens: Optional[int] = Field(
        default=None,
        description="Maximum total tokens budget for the entire query context (entities + relations + chunks + system prompt).",
        ge=1,
        le=MAX_QUERY_TOKEN_BUDGET,
    )

    hl_keywords: list[str] = Field(
        default_factory=list,
        max_length=MAX_KEYWORDS_PER_LIST,
        description="List of high-level keywords to prioritize in retrieval. Leave empty to use the LLM to generate the keywords.",
    )

    ll_keywords: list[str] = Field(
        default_factory=list,
        max_length=MAX_KEYWORDS_PER_LIST,
        description="List of low-level keywords to refine retrieval focus. Leave empty to use the LLM to generate the keywords.",
    )

    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="History messages are only sent to LLM for context, not used for retrieval. Format: [{'role': 'user/assistant', 'content': 'message'}].",
    )

    user_prompt: Optional[str] = Field(
        default=None,
        max_length=MAX_QUERY_CHARS,
        description=(
            "Additional instructions for the LLM, injected into the "
            "'Additional Instructions' section of the answer prompt. It does not "
            "affect retrieval. The server may prepend a global prefix to this "
            "value; if this field is empty the prefix alone becomes the "
            "instructions sent to the LLM. Set disable_user_prompt_prefix to "
            "opt out of the prefix."
        ),
    )

    disable_user_prompt_prefix: Optional[bool] = Field(
        default=None,
        description=(
            "If True, the server-side global prompt prefix is not prepended to "
            "user_prompt, leaving the client in full control of the final "
            "instruction text. This is the only way to suppress the prefix -- "
            "sending an empty user_prompt does not, it makes the prefix the "
            "whole instruction. The prefix itself is server configuration and "
            "cannot be read or replaced by a request. Default is False."
        ),
    )

    enable_rerank: Optional[bool] = Field(
        default=None,
        description="Enable reranking for retrieved text chunks. If True but no rerank model is configured, a warning will be issued. Default is True.",
    )

    include_references: Optional[bool] = Field(
        default=True,
        description="If True, includes reference list in responses. Affects /query and /query/stream endpoints. /query/data always includes references.",
    )

    include_chunk_content: Optional[bool] = Field(
        default=False,
        description="If True, includes actual chunk text content in references. Only applies when include_references=True. Useful for evaluation and debugging.",
    )

    include_progress: Optional[bool] = Field(
        default=False,
        description="If True, emits retrieval pipeline progress events (e.g. "
        "'extracting_keywords') before the response chunks and appends a final "
        "response_time metadata line. "
        "Only applies to /query/stream. When False (default), the stream "
        "preserves the original protocol order: references first, then "
        "response chunks — ensuring backward compatibility for existing clients.",
    )

    stream: Optional[bool] = Field(
        default=None,
        description="If True, enables streaming output. Defaults to False for /query, True for /query/stream.",
    )

    @field_validator("query", mode="after")
    @classmethod
    def query_strip_after(cls, query: str) -> str:
        # Mode-independent: `bypass` is exempt from the RAG minimum below, not
        # from having to carry a prompt at all. `min_length` on the field would
        # run before the strip and let "   " through.
        return validate_query_not_empty(query)

    @field_validator("hl_keywords", "ll_keywords", mode="after")
    @classmethod
    def keywords_length_check(cls, keywords: list[str]) -> list[str]:
        for keyword in keywords:
            if len(keyword) > MAX_KEYWORD_CHARS:
                raise ValueError(
                    f"each keyword must be at most {MAX_KEYWORD_CHARS} characters"
                )
        return keywords

    @field_validator("conversation_history", mode="after")
    @classmethod
    def conversation_history_role_check(
        cls, conversation_history: List[Dict[str, Any]] | None
    ) -> List[Dict[str, Any]] | None:
        if conversation_history is None:
            return None
        if len(conversation_history) > MAX_MESSAGES_PER_REQUEST:
            raise ValueError(
                f"conversation_history must hold at most {MAX_MESSAGES_PER_REQUEST} messages"
            )
        for msg in conversation_history:
            if "role" not in msg:
                raise ValueError("Each message must have a 'role' key.")
            if not isinstance(msg["role"], str) or not msg["role"].strip():
                raise ValueError("Each message 'role' must be a non-empty string.")
            if len(msg["role"]) > MAX_ROLE_CHARS:
                raise ValueError(
                    f"Each message 'role' must be at most {MAX_ROLE_CHARS} characters."
                )
            if "content" not in msg:
                raise ValueError("Each message must have a 'content' key.")
            if not isinstance(msg["content"], str):
                raise ValueError("Each message 'content' must be a string.")
            if len(msg["content"]) > MAX_MESSAGE_CHARS:
                raise ValueError(
                    f"Each message 'content' must be at most {MAX_MESSAGE_CHARS} characters."
                )

        # Extra keys stay allowed on purpose — clients following the OpenAI
        # convention send 'name' or 'tool_call_id'. The aggregate validator
        # counts the serialized whole, so those fields consume request budget
        # without being forbidden outright.
        return conversation_history

    @model_validator(mode="after")
    def enforce_rag_query_minimum(self) -> "QueryRequest":
        """Hold RAG queries to a length that can actually retrieve something.

        `bypass` goes straight to the LLM without touching the index, so the
        retrieval-shaped minimum does not apply to it; the non-empty check on
        the field does.
        """
        if self.mode != "bypass":
            validate_rag_query(self.query)
        return self

    @model_validator(mode="after")
    def bound_aggregate_text(self) -> "QueryRequest":
        """Bound all model-facing request input in one request.

        Per-field limits are not a bound on their own: the same payload is
        trivially rebuilt out of fields that are each individually legal. Every
        history dict is forwarded verbatim, so count its serialized form rather
        than only the ``content`` key.
        """
        # The server-side user_prompt_prefix is deliberately NOT counted: it is
        # operator configuration, not client-supplied text, and folding it in
        # would make previously-legal requests start failing the moment an
        # operator edits USER_PROMPT_PREFIX.
        total = count_conversation_input_chars(
            self.query,
            self.user_prompt,
            self.conversation_history,
        )
        if total > MAX_REQUEST_TEXT_CHARS:
            raise ValueError(
                f"total request text is {total} characters, over the "
                f"{MAX_REQUEST_TEXT_CHARS} character limit"
            )
        return self

    def to_query_params(self, is_stream: bool) -> "QueryParam":
        """Converts a QueryRequest instance into a QueryParam instance."""
        # Use Pydantic's `.model_dump(exclude_none=True)` to remove None values automatically
        # Exclude API-level parameters that don't belong in QueryParam
        request_data = self.model_dump(
            exclude_none=True,
            exclude={"query", "include_chunk_content", "include_progress"},
        )

        # Ensure `mode` and `stream` are set explicitly
        param = QueryParam(**request_data)
        param.stream = is_stream
        return param


class ReferenceItem(BaseModel):
    """A single reference item in query responses."""

    reference_id: str = Field(description="Unique reference identifier")
    file_path: str = Field(description="Path to the source file")
    content: Optional[List[str]] = Field(
        default=None,
        description="List of chunk contents from this file (only present when include_chunk_content=True)",
    )


class QueryResponse(BaseModel):
    response: str = Field(
        description="The generated response",
    )
    references: Optional[List[ReferenceItem]] = Field(
        default=None,
        description="Reference list (Disabled when include_references=False, /query/data always includes references.)",
    )
    response_time: Optional[float] = Field(
        default=None,
        description="Total server-side processing time in seconds (retrieval + LLM generation)",
    )
    llm_generated: bool = Field(
        default=True,
        description=(
            "Whether the answering LLM actually wrote this response. False when "
            "the query returned text produced without calling it: the canned "
            "no-context reply, or the only_need_context / only_need_prompt debug "
            "output. Clients that label machine-written text cannot recover this "
            "from the response text alone."
        ),
    )
    skills: List[str] = Field(
        default_factory=list,
        description="Skills selected and executed for this query.",
    )
    skill_plan: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Public skill routing metadata; raw planner output is omitted.",
    )


class QueryDataResponse(BaseModel):
    status: str = Field(description="Query execution status")
    message: str = Field(description="Status message")
    data: Dict[str, Any] = Field(
        description="Query result data containing entities, relationships, chunks, and references"
    )
    metadata: Dict[str, Any] = Field(
        description="Query metadata including mode, keywords, and processing information"
    )


class StreamChunkResponse(BaseModel):
    """Response model for streaming chunks in NDJSON format.

    Default stream order (``include_progress=False``):
    1. ``references`` — the reference list (only when
       ``include_references=True``), emitted once as the **first** line.
    2. ``response`` — LLM response content chunks (streaming) or the
       complete response (non-streaming).
    3. ``error`` — error message if processing fails.

    When the client opts in via ``include_progress=True``:
    ``progress`` lines are emitted **before** ``references``, so clients
    that depend on ``references`` being the first line should not enable
    ``include_progress``. A final ``response_time`` metadata line is also
    emitted after the response completes.
    """

    references: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Reference list (only in first chunk when include_references=True)",
    )
    response: Optional[str] = Field(
        default=None, description="Response content chunk or complete response"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if processing fails"
    )
    progress: Optional[str] = Field(
        default=None,
        description="Retrieval pipeline step identifier (e.g. 'extracting_keywords'); only emitted when include_progress=True",
    )
    response_time: Optional[float] = Field(
        default=None,
        description="Total server-side processing time in seconds (final metadata line when include_progress=True)",
    )
    llm_generated: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the answering LLM wrote the content on this line. Carried "
            "only on a complete (non-streamed) response line — the canned "
            "no-context reply and the only_need_context / only_need_prompt debug "
            "output are the cases that report False. Streamed response chunks "
            "always come from the LLM and carry no flag."
        ),
    )








async def _prepare_skill_routing(rag, request, param):
    if request.only_need_context or request.only_need_prompt:
        if param.mode == "bypass":
            param.mode = "mix"
        return None, {"ok": True, "skills": [], "engine": "native_rag_debug",
                      "reason": "原生RAG上下文/提示词查询"}
    return await EveQueryAgent(rag, request, param).run()


async def _query_with_skill_sources(rag, query, *, param, system_prompt, skill_plan, **kwargs):
    if skill_plan.get("_direct_response"):
        return {"status": "success", "data": {}, "llm_response": {
            "content": skill_plan["_direct_response"], "is_streaming": False,
            "response_iterator": None, "llm_generated": False}}
    result = await rag.aquery_llm(query, param=param, system_prompt=system_prompt, **kwargs)
    source_data = skill_plan.get("_rag_source_data", {})
    if source_data.get("references") or source_data.get("chunks"):
        data = result.setdefault("data", {})
        if not data.get("references"):
            data.update(source_data)
    return result








def _cited_source_footer(text: str, references: list[dict[str, Any]]) -> str:
    """Render verified citation filenames for clients that ignore reference events."""
    cited = set(re.findall(r"\[(?:reference_id\s*:\s*)?(\d+)\]", text or ""))
    lines = []
    seen = set()
    for ref in references:
        ref_id = str(ref.get("reference_id", ""))
        path = str(ref.get("file_path") or "")
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        name = " ".join(name.split()).replace("`", "'")
        if ref_id not in cited or not name or ref_id in seen:
            continue
        seen.add(ref_id)
        if name not in text:
            lines.append(f"- [{ref_id}] `{name}`")
    return "\n\n参考来源：\n" + "\n".join(lines) if lines else ""


def _public_skill_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Expose routing decisions without returning the planner's raw output."""
    return {
        "ok": bool(plan.get("ok", False)),
        "skills": list(plan.get("skills", []) or []),
        "reason": str(plan.get("reason", "") or ""),
        "engine": plan.get("engine", "langgraph"),
        "stages": list(plan.get("stages", [])),
        "skills_source": plan.get("skills_source"),
        "revision_count": int(plan.get("revision_count", 0)),
        "fitting_ids": list(plan.get("fitting_ids", [])),
        "simulation_count": int(plan.get("simulation_count", 0)),
        "ready_count": int(plan.get("ready_count", 0)),
        "stop_reason": str(plan.get("stop_reason", "")),
    }




def create_query_routes(rag, api_key: Optional[str] = None, top_k: int = 60):
    # Fresh router per call. A module-level instance would accumulate
    # duplicate routes when the factory is invoked more than once in the
    # same process (e.g. across tests), which triggers FastAPI's
    # "Duplicate Operation ID" warnings.
    router = APIRouter(tags=["query"])

    combined_auth = get_combined_auth_dependency(api_key)
    query_queue: asyncio.Queue[str] | None = None
    query_queue_workers: list[asyncio.Task[Any]] = []
    query_jobs: dict[str, dict[str, Any]] = {}

    def _query_debug_enabled() -> bool:
        return os.getenv("QUERY_DEBUG_AUDIT", "").casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _query_debug_path() -> Path:
        return Path(
            os.getenv(
                "QUERY_DEBUG_AUDIT_FILE",
                "logs/query_debug_audit.jsonl",
            )
        )

    def _public_request_payload(request: QueryRequest) -> dict[str, Any]:
        payload = request.model_dump(mode="json")
        # The API key/JWT never live in the Pydantic body, but keep this helper
        # body-only on purpose so future header changes cannot leak credentials.
        return payload

    def _audit_query_event(
        event: str,
        *,
        endpoint: str,
        request_obj: QueryRequest | None = None,
        http_request: Request | None = None,
        job_id: str | None = None,
        state: str | None = None,
        skill_plan: dict[str, Any] | None = None,
        result: Any = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not _query_debug_enabled():
            return
        record: dict[str, Any] = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": event,
            "endpoint": endpoint,
        }
        if job_id:
            record["job_id"] = job_id
        if state:
            record["state"] = state
        if http_request is not None and http_request.client is not None:
            record["client"] = {
                "host": http_request.client.host,
                "port": http_request.client.port,
            }
        if request_obj is not None:
            record["request"] = _public_request_payload(request_obj)
        if skill_plan is not None:
            record["skill_plan"] = _public_skill_plan(skill_plan)
        if isinstance(result, dict):
            record["result_summary"] = {
                "skills": result.get("skills", []),
                "skill_plan": result.get("skill_plan"),
                "response_time": result.get("response_time"),
                "llm_generated": result.get("llm_generated"),
            }
        if error:
            record["error"] = error
        if extra:
            record.update(extra)
        try:
            path = _query_debug_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("Failed to write query debug audit log", exc_info=True)

    def _strip_internal_reasoning(text: str) -> str:
        """Keep internal reasoning tags out of the public API response."""
        cleaned = re.sub(
            r"<(?:think|analysis)\b[^>]*>.*?</(?:think|analysis)\s*>",
            "",
            text or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        return cleaned.strip()

    def _queue_workers_count() -> int:
        try:
            configured = int(os.getenv("QUERY_QUEUE_WORKERS", "4"))
        except (TypeError, ValueError):
            configured = 4
        return max(1, min(configured, 16))

    def _queue_max_size() -> int:
        try:
            configured = int(os.getenv("QUERY_QUEUE_MAX_SIZE", "200"))
        except (TypeError, ValueError):
            configured = 200
        return max(1, min(configured, 10000))

    def _queue_retries() -> int:
        try:
            configured = int(os.getenv("QUERY_QUEUE_RETRIES", "2"))
        except (TypeError, ValueError):
            configured = 2
        return max(0, min(configured, 5))


    def _cleanup_query_jobs() -> None:
        try:
            retention = int(os.getenv("QUERY_QUEUE_RETENTION_SECONDS", "3600"))
        except (TypeError, ValueError):
            retention = 3600
        cutoff = time.time() - max(60, retention)
        stale = [
            job_id
            for job_id, job in query_jobs.items()
            if job.get("status") in {"completed", "failed"}
            and job.get("finished_at", 0) < cutoff
        ]
        for job_id in stale:
            query_jobs.pop(job_id, None)

    async def _query_queue_worker() -> None:
        assert query_queue is not None
        while True:
            job_id = await query_queue.get()
            job = query_jobs.get(job_id)
            if job is None:
                query_queue.task_done()
                continue
            job["status"] = "running"
            job["started_at"] = time.time()
            attempts = _queue_retries() + 1
            try:
                for attempt in range(1, attempts + 1):
                    try:
                        job["attempt"] = attempt
                        result = await query_text(job["request"])
                        if hasattr(result, "model_dump"):
                            result = result.model_dump()
                        job["result"] = result
                        job["status"] = "completed"
                        _audit_query_event(
                            "queued_completed",
                            endpoint="/query/queued",
                            request_obj=job.get("request"),
                            job_id=job_id,
                            state="completed",
                            result=result,
                            extra={"attempt": attempt},
                        )
                        break
                    except Exception as exc:
                        job["error"] = str(exc)
                        if attempt >= attempts:
                            job["status"] = "failed"
                            logger.error("Queued query failed: %s", exc, exc_info=True)
                            _audit_query_event(
                                "queued_failed",
                                endpoint="/query/queued",
                                request_obj=job.get("request"),
                                job_id=job_id,
                                state="failed",
                                error=str(exc),
                                extra={"attempt": attempt},
                            )
                            break
                        job["status"] = "retrying"
                        await asyncio.sleep(min(2 ** (attempt - 1), 8))
            finally:
                job["finished_at"] = time.time()
                query_queue.task_done()

    def _ensure_query_queue() -> None:
        nonlocal query_queue
        if query_queue is None:
            query_queue = asyncio.Queue(maxsize=_queue_max_size())
        alive = [worker for worker in query_queue_workers if not worker.done()]
        query_queue_workers[:] = alive
        while len(query_queue_workers) < _queue_workers_count():
            query_queue_workers.append(asyncio.create_task(_query_queue_worker()))

    def _public_job(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
        response: dict[str, Any] = {
            "status": "success",
            "job_id": job_id,
            "state": job["status"],
            "queue_size": query_queue.qsize() if query_queue else 0,
        }
        if job["status"] == "completed":
            response["result"] = job.get("result")
        elif job["status"] == "failed":
            response["error"] = job.get("error", "queued query failed")
        return response

    @router.get(
        "/skills",
        dependencies=[Depends(combined_auth)],
        summary="List query skills",
    )
    async def query_skills():
        return {
            "status": "success",
            "skills": [
                {"name": name, "description": SKILL_DESCRIPTIONS[name]}
                for name in SKILL_NAMES
            ],
        }

    @router.get(
        "/skills/plan",
        dependencies=[Depends(combined_auth)],
        summary="Let the query LLM select skills",
    )
    async def query_skill_plan(
        q: str = Query(..., min_length=1, description="User query"),
    ):
        try:
            plan = await plan_skills(rag, q)
            public_plan = {
                key: value for key, value in plan.items() if key != "raw"
            }
            return {
                "status": "success",
                "query": q,
                "plan": public_plan,
            }
        except Exception as e:
            logger.error("Query skill planning failed: %s", e, exc_info=True)
            raise internal_server_error(e)

    @router.get(
        "/skills/neo4j",
        dependencies=[Depends(combined_auth)],
        summary="Query the Neo4j knowledge graph skill",
    )
    async def query_neo4j_skill(
        q: str = Query(..., min_length=1, description="Graph query"),
        limit: int = Query(20, ge=1, le=100),
    ):
        try:
            return {
                "status": "success",
                **(await query_neo4j(rag, q, limit)),
            }
        except Exception as e:
            logger.error("Neo4j skill query failed: %s", e, exc_info=True)
            raise internal_server_error(e)

    @router.get(
        "/skills/milvus",
        dependencies=[Depends(combined_auth)],
        summary="Query the Milvus vector search skill",
    )
    async def query_milvus_skill(
        q: str = Query(..., min_length=1, description="Semantic vector query"),
        limit: int = Query(20, ge=1, le=100),
    ):
        try:
            return {
                "status": "success",
                **(await query_milvus(rag, q, limit)),
            }
        except Exception as e:
            logger.error("Milvus skill query failed: %s", e, exc_info=True)
            raise internal_server_error(e)

    @router.get(
        "/eve/fittings",
        dependencies=[Depends(combined_auth)],
        summary="Query all EVE fitting schemes for a ship",
    )
    async def eve_fittings(
        ship: str = Query(..., min_length=1, description="Ship name in Chinese or English"),
    ):
        """Return every fitting scheme matching the requested ship name."""
        try:
            fittings = await run_eve_db_call(find_all_fittings, ship.strip())
            return {
                "status": "success",
                "ship": ship.strip(),
                "count": len(fittings),
                "fittings": fittings,
            }
        except Exception as e:
            logger.error("EVE fitting API lookup failed: %s", e, exc_info=True)
            raise internal_server_error(e)

    @router.post(
        "/eve/fitting/simulate",
        dependencies=[Depends(combined_auth)],
        summary="Simulate and validate an EVE fitting with Pyfa/Eos",
    )
    async def eve_fitting_simulate(payload: Dict[str, Any]):
        """Calculate a fit without importing the Pyfa desktop GUI."""
        try:
            result = await run_eve_db_call(simulate_fitting, payload)
            return {"status": "success", "result": result}
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error("Pyfa fitting simulation failed: %s", e, exc_info=True)
            raise internal_server_error(e)

    @router.get(
        "/eve/characters",
        dependencies=[Depends(combined_auth)],
        summary="List EVE characters with stored ESI authorization",
    )
    async def eve_characters():
        try:
            characters = await run_eve_db_call(list_authorized_characters)
            return {
                "status": "success",
                "count": len(characters),
                "characters": characters,
            }
        except Exception as e:
            logger.error("EVE character list failed: %s", e, exc_info=True)
            raise internal_server_error(e)

    @router.get(
        "/eve/characters/{character_ref}/skills",
        dependencies=[Depends(combined_auth)],
        summary="Read an authorized character's ESI skills",
    )
    async def eve_character_skills(
        character_ref: str,
        include_queue: bool = Query(True),
        include_attributes: bool = Query(False),
        force_refresh: bool = Query(False),
    ):
        try:
            result = await run_eve_db_call(
                get_character_skill_data,
                character_ref,
                include_queue,
                include_attributes,
                force_refresh,
            )
            return {"status": "success", **result}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error("EVE character skill lookup failed: %s", e, exc_info=True)
            raise internal_server_error(e)

    @router.get(
        "/eve/static-data",
        dependencies=[Depends(combined_auth)],
        summary="Query EVE SDE static data",
    )
    async def eve_static_data(
        q: str = Query("", description="Name, ID, or text to search"),
        dataset: Optional[str] = Query(None, description="JSONL dataset name, e.g. types"),
        record_key: Optional[str] = Query(None, description="Exact _key or record identifier"),
        limit: int = Query(20, ge=1, le=100),
    ):
        """Query any imported EVE SDE dataset."""
        try:
            rows = await run_eve_db_call(
                query_static_data, q, dataset, record_key, limit
            )
            return {
                "status": "success",
                "query": q,
                "dataset": dataset,
                "record_key": record_key,
                "count": len(rows),
                "results": rows,
            }
        except Exception as e:
            logger.error("EVE static data API lookup failed: %s", e, exc_info=True)
            raise internal_server_error(e)

    @router.get(
        "/eve/static-data/datasets",
        dependencies=[Depends(combined_auth)],
        summary="List imported EVE SDE datasets",
    )
    async def eve_static_data_datasets():
        try:
            names = await run_eve_db_call(dataset_names)
            return {"status": "success", "count": len(names), "datasets": names}
        except Exception as e:
            logger.error("EVE SDE dataset listing failed: %s", e, exc_info=True)
            raise internal_server_error(e)

    async def _eve_static_dataset_shortcut(
        dataset: str,
        q: str,
        record_key: Optional[str],
        limit: int,
    ):
        rows = await run_eve_db_call(
            query_static_data, q, dataset, record_key, limit
        )
        return {
            "status": "success",
            "dataset": dataset,
            "query": q,
            "record_key": record_key,
            "count": len(rows),
            "results": rows,
        }

    @router.get(
        "/eve/types",
        dependencies=[Depends(combined_auth)],
        summary="Query EVE item and ship types",
    )
    async def eve_types(
        q: str = Query("", description="Chinese or English type name"),
        record_key: Optional[str] = Query(None, description="typeID"),
        limit: int = Query(20, ge=1, le=100),
    ):
        return await _eve_static_dataset_shortcut("types", q, record_key, limit)

    @router.get(
        "/eve/blueprints",
        dependencies=[Depends(combined_auth)],
        summary="Query EVE blueprints",
    )
    async def eve_blueprints(
        q: str = Query("", description="Blueprint name or text"),
        record_key: Optional[str] = Query(None, description="blueprintTypeID"),
        limit: int = Query(20, ge=1, le=100),
    ):
        return await _eve_static_dataset_shortcut("blueprints", q, record_key, limit)

    @router.get(
        "/eve/systems",
        dependencies=[Depends(combined_auth)],
        summary="Query EVE solar systems",
    )
    async def eve_systems(
        q: str = Query("", description="Solar system name"),
        record_key: Optional[str] = Query(None, description="solarSystemID"),
        limit: int = Query(20, ge=1, le=100),
    ):
        return await _eve_static_dataset_shortcut(
            "mapsolarsystems", q, record_key, limit
        )

    @router.post(
        "/query",
        response_model=QueryResponse,
        dependencies=[Depends(combined_auth)],
        responses={
            200: {
                "description": "Successful RAG query response",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "response": {
                                    "type": "string",
                                    "description": "The generated response from the RAG system",
                                },
                                "references": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "reference_id": {"type": "string"},
                                            "file_path": {"type": "string"},
                                            "content": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "List of chunk contents from this file (only included when include_chunk_content=True)",
                                            },
                                        },
                                    },
                                    "description": "Reference list (only included when include_references=True)",
                                },
                            },
                            "required": ["response"],
                        },
                        "examples": {
                            "with_references": {
                                "summary": "Response with references",
                                "description": "Example response when include_references=True",
                                "value": {
                                    "response": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.",
                                    "references": [
                                        {
                                            "reference_id": "1",
                                            "file_path": "/documents/ai_overview.pdf",
                                        },
                                        {
                                            "reference_id": "2",
                                            "file_path": "/documents/machine_learning.txt",
                                        },
                                    ],
                                },
                            },
                            "with_chunk_content": {
                                "summary": "Response with chunk content",
                                "description": "Example response when include_references=True and include_chunk_content=True. Note: content is an array of chunks from the same file.",
                                "value": {
                                    "response": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.",
                                    "references": [
                                        {
                                            "reference_id": "1",
                                            "file_path": "/documents/ai_overview.pdf",
                                            "content": [
                                                "Artificial Intelligence (AI) represents a transformative field in computer science focused on creating systems that can perform tasks requiring human-like intelligence. These tasks include learning from experience, understanding natural language, recognizing patterns, and making decisions.",
                                                "AI systems can be categorized into narrow AI, which is designed for specific tasks, and general AI, which aims to match human cognitive abilities across a wide range of domains.",
                                            ],
                                        },
                                        {
                                            "reference_id": "2",
                                            "file_path": "/documents/machine_learning.txt",
                                            "content": [
                                                "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It focuses on the development of algorithms that can access data and use it to learn for themselves."
                                            ],
                                        },
                                    ],
                                },
                            },
                            "without_references": {
                                "summary": "Response without references",
                                "description": "Example response when include_references=False",
                                "value": {
                                    "response": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving."
                                },
                            },
                            "different_modes": {
                                "summary": "Different query modes",
                                "description": "Examples of responses from different query modes",
                                "value": {
                                    "local_mode": "Focuses on specific entities and their relationships",
                                    "global_mode": "Provides broader context from relationship patterns",
                                    "hybrid_mode": "Combines local and global approaches",
                                    "naive_mode": "Simple vector similarity search",
                                    "mix_mode": "Integrates knowledge graph and vector retrieval",
                                },
                            },
                        },
                    }
                },
            },
            422: {
                "description": "Unprocessable Entity - Request validation failed",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "detail": {"type": "array", "items": {"type": "object"}}
                            },
                        },
                        "example": {
                            "detail": [
                                {
                                    "type": "value_error",
                                    "loc": ["body"],
                                    "msg": (
                                        "Value error, RAG query is too short. Enter at "
                                        "least 3 English characters or an equivalent "
                                        "combination where each Chinese, Japanese or "
                                        "Korean character counts as 2."
                                    ),
                                }
                            ]
                        },
                    }
                },
            },
            500: {
                "description": "Internal Server Error - Query processing failed",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"detail": {"type": "string"}},
                        },
                        "example": {
                            "detail": "Failed to process query: LLM service unavailable"
                        },
                    }
                },
            },
        },
    )
    async def query_text(request: QueryRequest, http_request: Request = None):
        """
        Comprehensive RAG query endpoint with non-streaming response. Parameter "stream" is ignored.

        **Query Modes:**
        - **local**: Focuses on specific entities and their direct relationships
        - **global**: Analyzes broader patterns and relationships across the knowledge graph
        - **hybrid**: Combines local and global approaches for comprehensive results
        - **naive**: Simple vector similarity search without knowledge graph
        - **mix**: Integrates knowledge graph retrieval with vector search (recommended)
        - **bypass**: Direct LLM query without knowledge retrieval

        conversation_history parameteris sent to LLM only, does not affect retrieval results.

        **Usage Examples:**

        Basic query:
        ```json
        {
            "query": "What is machine learning?",
            "mode": "mix"
        }
        ```

        Bypass initial LLM call by providing high-level and low-level keywords:
        ```json
        {
            "query": "What is Retrieval-Augmented-Generation?",
            "hl_keywords": ["machine learning", "information retrieval", "natural language processing"],
            "ll_keywords": ["retrieval augmented generation", "RAG", "knowledge base"],
            "mode": "mix"
        }
        ```

        Advanced query with references:
        ```json
        {
            "query": "Explain neural networks",
            "mode": "hybrid",
            "include_references": true,
            "response_type": "Multiple Paragraphs",
            "top_k": 10
        }
        ```

        Conversation with history:
        ```json
        {
            "query": "Can you give me more details?",
            "conversation_history": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is artificial intelligence..."}
            ]
        }
        ```

        Args:
            request (QueryRequest): The request object containing query parameters:
                - **query**: The question or prompt to process. Must not be empty. RAG modes additionally require an English-equivalent length of at least 3; each Chinese, Japanese or Korean character counts as 2. Bypass mode is exempt from the minimum, not from being non-empty.
                - **mode**: Query strategy - "mix" recommended for best results
                - **include_references**: Whether to include source citations
                - **response_type**: Format preference (e.g., "Multiple Paragraphs")
                - **top_k**: Number of top entities/relations to retrieve
                - **conversation_history**: Previous dialogue context
                - **max_total_tokens**: Token budget for the entire response

        Returns:
            QueryResponse: JSON response containing:
                - **response**: The generated answer to your query
                - **references**: Source citations (if include_references=True)

        Raises:
            HTTPException:
                - 422: Request validation failed (e.g., query empty or too short)
                - 500: Internal processing error (e.g., LLM service unavailable)
        """
        try:
            _audit_query_event(
                "query_received",
                endpoint="/query",
                request_obj=request,
                http_request=http_request,
            )
            param = request.to_query_params(
                False
            )  # Ensure stream=False for non-streaming endpoint
            # Force stream=False for /query endpoint regardless of include_references setting
            param.stream = False
            start_time = time.perf_counter()
            skill_system_prompt, skill_plan = await _prepare_skill_routing(
                rag, request, param
            )
            # Unified approach: always use aquery_llm for both cases
            result = await _query_with_skill_sources(
                rag,
                request.query,
                param=param,
                system_prompt=skill_system_prompt,
                skill_plan=skill_plan,
            )
            response_time = round(time.perf_counter() - start_time, 3)

            # Extract LLM response and references from unified result
            llm_response = result.get("llm_response", {})
            data = result.get("data", {})
            references = data.get("references", [])

            # Get the non-streaming response content
            response_content = llm_response.get("content", "")
            if not (request.only_need_context or request.only_need_prompt) and llm_response.get("llm_generated", True):
                response_content = _strip_internal_reasoning(response_content)
                from eve_lightrag.eve_names import localize_text
                response_content = await run_eve_db_call(localize_text, response_content)
            # Whether the answering LLM wrote it. A missing flag means a result
            # shape that predates it, which only the LLM paths produce.
            llm_generated = bool(llm_response.get("llm_generated", True))
            if not response_content:
                response_content = "No relevant context found for the query."
                # Substituted placeholder: nothing was generated.
                llm_generated = False

            if request.include_references and llm_generated:
                response_content += _cited_source_footer(response_content, references)

            # Enrich references with chunk content if requested
            if request.include_references and request.include_chunk_content:
                chunks = data.get("chunks", [])
                # Create a mapping from reference_id to chunk content
                ref_id_to_content = {}
                for chunk in chunks:
                    ref_id = chunk.get("reference_id", "")
                    content = chunk.get("content", "")
                    if ref_id and content:
                        # Collect chunk content; join later to avoid quadratic string concatenation
                        ref_id_to_content.setdefault(ref_id, []).append(content)

                # Add content to references
                enriched_references = []
                for ref in references:
                    ref_copy = ref.copy()
                    ref_id = ref.get("reference_id", "")
                    if ref_id in ref_id_to_content:
                        # Keep content as a list of chunks (one file may have multiple chunks)
                        ref_copy["content"] = ref_id_to_content[ref_id]
                    enriched_references.append(ref_copy)
                references = enriched_references

            # Return response with or without references based on request
            if request.include_references:
                response = QueryResponse(
                    response=response_content,
                    references=references,
                    response_time=response_time,
                    llm_generated=llm_generated,
                    skills=skill_plan.get("skills", []),
                    skill_plan=_public_skill_plan(skill_plan),
                )
            else:
                response = QueryResponse(
                    response=response_content,
                    references=None,
                    response_time=response_time,
                    llm_generated=llm_generated,
                    skills=skill_plan.get("skills", []),
                    skill_plan=_public_skill_plan(skill_plan),
                )
            _audit_query_event(
                "query_completed",
                endpoint="/query",
                request_obj=request,
                http_request=http_request,
                state="completed",
                skill_plan=skill_plan,
                result=response.model_dump(mode="json"),
            )
            return response
        except Exception as e:
            _audit_query_event(
                "query_failed",
                endpoint="/query",
                request_obj=request,
                http_request=http_request,
                state="failed",
                error=str(e),
            )
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise internal_server_error(e)

    @router.post(
        "/query/queued",
        dependencies=[Depends(combined_auth)],
        status_code=202,
        summary="Submit a query to the bounded batch queue",
    )
    async def query_text_queued(request: QueryRequest, http_request: Request):
        """Queue a query and return immediately with a pollable job ID."""
        _ensure_query_queue()
        _cleanup_query_jobs()
        assert query_queue is not None
        if query_queue.full():
            raise HTTPException(status_code=503, detail="查询队列已满，请稍后重试")
        job_id = uuid.uuid4().hex
        query_jobs[job_id] = {
            "status": "queued",
            "request": request,
            "created_at": time.time(),
        }
        await query_queue.put(job_id)
        _audit_query_event(
            "queued_submitted",
            endpoint="/query/queued",
            request_obj=request,
            http_request=http_request,
            job_id=job_id,
            state="queued",
            extra={"queue_size": query_queue.qsize()},
        )
        return _public_job(job_id, query_jobs[job_id])

    @router.get(
        "/query/queued/{job_id}",
        dependencies=[Depends(combined_auth)],
        summary="Get a queued query result",
    )
    async def query_text_queued_status(job_id: str):
        job = query_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="查询任务不存在或已过期")
        return _public_job(job_id, job)

    def _build_stream_generator(
        *,
        result: dict[str, Any],
        include_references: bool,
        include_chunk_content: bool,
        include_response_time: bool,
        start_time: float,
        skill_plan: Optional[Dict[str, Any]] = None,
    ):
        """Shared async generator that yields NDJSON lines for streaming responses.

        Used by ``/query/stream`` to format NDJSON output with consistent
        error-handling behaviour. When ``include_response_time`` is enabled,
        a final metadata line is emitted after all content so opted-in clients
        can display the total server-side processing duration.
        """

        async def _generate():
            references = result.get("data", {}).get("references", [])
            llm_response = result.get("llm_response", {})

            # Enrich references with chunk content if requested
            if include_references and include_chunk_content:
                data = result.get("data", {})
                chunks = data.get("chunks", [])
                ref_id_to_content: dict[str, list[str]] = {}
                for chunk in chunks:
                    ref_id = chunk.get("reference_id", "")
                    content = chunk.get("content", "")
                    if ref_id and content:
                        ref_id_to_content.setdefault(ref_id, []).append(content)

                enriched_references = []
                for ref in references:
                    ref_copy = ref.copy()
                    ref_id = ref.get("reference_id", "")
                    if ref_id in ref_id_to_content:
                        ref_copy["content"] = ref_id_to_content[ref_id]
                    enriched_references.append(ref_copy)
                references = enriched_references

            if llm_response.get("is_streaming"):
                # Streaming: references first, then response chunks
                if include_references:
                    yield f"{json.dumps({'references': references})}\n"

                response_stream = llm_response.get("response_iterator")
                if response_stream:
                    try:
                        from eve_lightrag.eve_names import NameStream
                        name_stream = NameStream()
                        rendered_parts = []
                        async for chunk in response_stream:
                            if chunk:
                                translated = name_stream.feed(chunk)
                                if translated:
                                    rendered_parts.append(translated)
                                    yield f"{json.dumps({'response': translated})}\n"
                        translated = name_stream.feed(final=True)
                        if translated:
                            rendered_parts.append(translated)
                            yield f"{json.dumps({'response': translated})}\n"
                        if include_references:
                            footer = _cited_source_footer("".join(rendered_parts), references)
                            if footer:
                                yield f"{json.dumps({'response': footer})}\n"
                    except Exception as e:
                        logger.error(f"Streaming error: {str(e)}")
                        yield f"{json.dumps({'error': str(e)})}\n"
            else:
                # Non-streaming: complete response in one message
                response_content = llm_response.get("content", "")
                if llm_response.get("llm_generated", True):
                    from eve_lightrag.eve_names import localize_text
                    response_content = await run_eve_db_call(localize_text, response_content)
                llm_generated = bool(llm_response.get("llm_generated", True))
                if not response_content:
                    response_content = "No relevant context found for the query."
                    llm_generated = False

                if include_references and llm_generated:
                    response_content += _cited_source_footer(response_content, references)

                # The flag rides this line rather than a separate one: it
                # describes THIS content, and a client that ignores the key
                # still reads the response exactly as before. Streamed chunks
                # (the branch above) always come from the LLM, so they need no
                # flag.
                complete_response: dict[str, Any] = {
                    "response": response_content,
                    "llm_generated": llm_generated,
                }
                if include_references:
                    complete_response["references"] = references
                if skill_plan is not None:
                    complete_response["skills"] = skill_plan.get("skills", [])
                    complete_response["skill_plan"] = skill_plan

                yield f"{json.dumps(complete_response)}\n"

            if include_response_time:
                # Final metadata line: total server-side processing time
                # (retrieval + LLM generation) for opted-in clients.
                metadata = {
                    "response_time": round(time.perf_counter() - start_time, 3)
                }
                if skill_plan is not None:
                    metadata["skills"] = skill_plan.get("skills", [])
                    metadata["skill_plan"] = skill_plan
                yield f"{json.dumps(metadata)}\n"

        return _generate

    @router.post(
        "/query/stream",
        dependencies=[Depends(combined_auth)],
        responses={
            200: {
                "description": "Flexible RAG query response - format depends on stream parameter",
                "content": {
                    "application/x-ndjson": {
                        "schema": {
                            "type": "string",
                            "format": "ndjson",
                            "description": "Newline-delimited JSON (NDJSON) format used for both streaming and non-streaming responses. For streaming: multiple lines with separate JSON objects. For non-streaming: single line with complete JSON object.",
                            "example": '{"references": [{"reference_id": "1", "file_path": "/documents/ai.pdf"}]}\n{"response": "Artificial Intelligence is"}\n{"response": " a field of computer science"}\n{"response": " that focuses on creating intelligent machines."}',
                        },
                        "examples": {
                            "streaming_with_references": {
                                "summary": "Streaming mode with references (stream=true)",
                                "description": "Multiple NDJSON lines when stream=True and include_references=True. First line contains references, subsequent lines contain response chunks.",
                                "value": '{"references": [{"reference_id": "1", "file_path": "/documents/ai_overview.pdf"}, {"reference_id": "2", "file_path": "/documents/ml_basics.txt"}]}\n{"response": "Artificial Intelligence (AI) is a branch of computer science"}\n{"response": " that aims to create intelligent machines capable of performing"}\n{"response": " tasks that typically require human intelligence, such as learning,"}\n{"response": " reasoning, and problem-solving."}',
                            },
                            "streaming_with_chunk_content": {
                                "summary": "Streaming mode with chunk content (stream=true, include_chunk_content=true)",
                                "description": "Multiple NDJSON lines when stream=True, include_references=True, and include_chunk_content=True. First line contains references with content arrays (one file may have multiple chunks), subsequent lines contain response chunks.",
                                "value": '{"references": [{"reference_id": "1", "file_path": "/documents/ai_overview.pdf", "content": ["Artificial Intelligence (AI) represents a transformative field...", "AI systems can be categorized into narrow AI and general AI..."]}, {"reference_id": "2", "file_path": "/documents/ml_basics.txt", "content": ["Machine learning is a subset of AI that enables computers to learn..."]}]}\n{"response": "Artificial Intelligence (AI) is a branch of computer science"}\n{"response": " that aims to create intelligent machines capable of performing"}\n{"response": " tasks that typically require human intelligence."}',
                            },
                            "streaming_without_references": {
                                "summary": "Streaming mode without references (stream=true)",
                                "description": "Multiple NDJSON lines when stream=True and include_references=False. Only response chunks are sent.",
                                "value": '{"response": "Machine learning is a subset of artificial intelligence"}\n{"response": " that enables computers to learn and improve from experience"}\n{"response": " without being explicitly programmed for every task."}',
                            },
                            "non_streaming_with_references": {
                                "summary": "Non-streaming mode with references (stream=false)",
                                "description": "Single NDJSON line when stream=False and include_references=True. Complete response with references in one message.",
                                "value": '{"references": [{"reference_id": "1", "file_path": "/documents/neural_networks.pdf"}], "response": "Neural networks are computational models inspired by biological neural networks that consist of interconnected nodes (neurons) organized in layers. They are fundamental to deep learning and can learn complex patterns from data through training processes."}',
                            },
                            "non_streaming_without_references": {
                                "summary": "Non-streaming mode without references (stream=false)",
                                "description": "Single NDJSON line when stream=False and include_references=False. Complete response only.",
                                "value": '{"response": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence deep) to model and understand complex patterns in data. It has revolutionized fields like computer vision, natural language processing, and speech recognition."}',
                            },
                            "error_response": {
                                "summary": "Error during streaming",
                                "description": "Error handling in NDJSON format when an error occurs during processing.",
                                "value": '{"references": [{"reference_id": "1", "file_path": "/documents/ai.pdf"}]}\n{"response": "Artificial Intelligence is"}\n{"error": "LLM service temporarily unavailable"}',
                            },
                        },
                    }
                },
            },
            422: {
                "description": "Unprocessable Entity - Request validation failed",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "detail": {"type": "array", "items": {"type": "object"}}
                            },
                        },
                        "example": {
                            "detail": [
                                {
                                    "type": "value_error",
                                    "loc": ["body"],
                                    "msg": (
                                        "Value error, RAG query is too short. Enter at "
                                        "least 3 English characters or an equivalent "
                                        "combination where each Chinese, Japanese or "
                                        "Korean character counts as 2."
                                    ),
                                }
                            ]
                        },
                    }
                },
            },
            500: {
                "description": "Internal Server Error - Query processing failed",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"detail": {"type": "string"}},
                        },
                        "example": {
                            "detail": "Failed to process streaming query: Knowledge graph unavailable"
                        },
                    }
                },
            },
        },
    )
    async def query_text_stream(request: QueryRequest):
        """
        Advanced RAG query endpoint with flexible streaming response.

        This endpoint provides the most flexible querying experience, supporting both real-time streaming
        and complete response delivery based on your integration needs.

        **Response Modes:**
        - Real-time response delivery as content is generated
        - NDJSON format: each line is a separate JSON object
        - Default order (``include_progress=False``):
          - First line: `{"references": [...]}` (if include_references=True)
          - Subsequent lines: `{"response": "content chunk"}`
          - Error handling: `{"error": "error message"}`
        - With ``include_progress=True`` (opt-in):
          - Progress lines: `{"progress": "step_name"}` emitted **before** references
          - Then references, response chunks, and errors as above
          - Final line: `{"response_time": 1.234}`
          - Clients that depend on references being the first line must not enable ``include_progress``

        > If stream parameter is False, or the query hit LLM cache, complete response delivered in a single streaming message.

        **Response Format Details**
        - **Content-Type**: `application/x-ndjson` (Newline-Delimited JSON)
        - **Structure**: Each line is an independent, valid JSON object
        - **Parsing**: Process line-by-line, each line is self-contained
        - **Headers**: Includes cache control and connection management

        **Query Modes (same as /query endpoint)**
        - **local**: Entity-focused retrieval with direct relationships
        - **global**: Pattern analysis across the knowledge graph
        - **hybrid**: Combined local and global strategies
        - **naive**: Vector similarity search only
        - **mix**: Integrated knowledge graph + vector retrieval (recommended)
        - **bypass**: Direct LLM query without knowledge retrieval

        conversation_history parameteris sent to LLM only, does not affect retrieval results.

        **Usage Examples**

        Real-time streaming query:
        ```json
        {
            "query": "Explain machine learning algorithms",
            "mode": "mix",
            "stream": true,
            "include_references": true
        }
        ```

        Bypass initial LLM call by providing high-level and low-level keywords:
        ```json
        {
            "query": "What is Retrieval-Augmented-Generation?",
            "hl_keywords": ["machine learning", "information retrieval", "natural language processing"],
            "ll_keywords": ["retrieval augmented generation", "RAG", "knowledge base"],
            "mode": "mix"
        }
        ```

        Complete response query:
        ```json
        {
            "query": "What is deep learning?",
            "mode": "hybrid",
            "stream": false,
            "response_type": "Multiple Paragraphs"
        }
        ```

        Streaming with retrieval progress (opt-in):
        ```json
        {
            "query": "Explain neural networks",
            "mode": "mix",
            "stream": true,
            "include_progress": true
        }
        ```
        Progress lines (`{"progress": "extracting_keywords"}`, etc.) are emitted
        before references and response chunks, letting the client show live
        pipeline status. A final `{"response_time": 1.234}` metadata line follows
        the response. Omit ``include_progress`` (or set it to ``false``) to keep
        the original protocol shape with no progress or timing metadata lines.

        Conversation with context:
        ```json
        {
            "query": "Can you elaborate on that?",
            "stream": true,
            "conversation_history": [
                {"role": "user", "content": "What is neural network?"},
                {"role": "assistant", "content": "A neural network is..."}
            ]
        }
        ```

        **Response Processing:**

        ```python
        async for line in response.iter_lines():
            data = json.loads(line)
            if "references" in data:
                # Handle references (first message)
                references = data["references"]
            if "response" in data:
                # Handle content chunk
                content_chunk = data["response"]
            if "error" in data:
                # Handle error
                error_message = data["error"]
        ```

        **Error Handling:**
        - Streaming errors are delivered as `{"error": "message"}` lines
        - Non-streaming errors raise HTTP exceptions
        - Partial responses may be delivered before errors in streaming mode
        - Always check for error objects when processing streaming responses

        Args:
            request (QueryRequest): The request object containing query parameters:
                - **query**: The question or prompt to process. Must not be empty. RAG modes additionally require an English-equivalent length of at least 3; each Chinese, Japanese or Korean character counts as 2. Bypass mode is exempt from the minimum, not from being non-empty.
                - **mode**: Query strategy - "mix" recommended for best results
                - **stream**: Enable streaming (True) or complete response (False)
                - **include_references**: Whether to include source citations
                - **include_progress**: If True, emit retrieval progress events before references and a final response_time metadata line (default: False)
                - **response_type**: Format preference (e.g., "Multiple Paragraphs")
                - **top_k**: Number of top entities/relations to retrieve
                - **conversation_history**: Previous dialogue context for multi-turn conversations
                - **max_total_tokens**: Token budget for the entire response

        Returns:
            StreamingResponse: NDJSON streaming response containing:
                - **Streaming mode**: Multiple JSON objects, one per line
                  - Progress objects (only if include_progress=True): `{"progress": "step_name"}`
                  - References object (if requested): `{"references": [...]}`
                  - Content chunks: `{"response": "chunk content"}`
                  - Error objects: `{"error": "error message"}`
                  - Final timing object (only if include_progress=True): `{"response_time": 1.234}`
                - **Non-streaming mode**: Single JSON object
                  - Complete response: `{"references": [...], "response": "complete content"}`

        Raises:
            HTTPException:
                - 422: Request validation failed (e.g., query empty or too short, invalid mode)
                - 500: Internal processing error (e.g., LLM service unavailable)

        Note:
            This endpoint is ideal for applications requiring flexible response delivery.
            Use streaming mode for real-time interfaces and non-streaming for batch processing.
        """
        try:
            # Use the stream parameter from the request, defaulting to True if not specified
            stream_mode = request.stream if request.stream is not None else True
            param = request.to_query_params(stream_mode)
            start_time = time.perf_counter()
            skill_system_prompt, skill_plan = await _prepare_skill_routing(
                rag, request, param
            )

            from fastapi.responses import StreamingResponse

            # When the client opts in via include_progress, run aquery_llm as a
            # background task so progress events can be interleaved into the
            # NDJSON stream before the response chunks. When include_progress
            # is False (default), use the original blocking path that preserves
            # the exact protocol order: references → response chunks → time.
            include_progress = request.include_progress or False

            if include_progress:
                progress_queue: asyncio.Queue = asyncio.Queue()
                # Sentinel enqueued once the query task settles, so the generator
                # can drain progress events with a blocking get() instead of
                # polling task.done() on a 0.1s timeout (which could also drop an
                # event whenever the timeout raced the get()).
                done_sentinel = object()

                async def progress_callback(event: str):
                    await progress_queue.put(event)

                async def run_query():
                    # Wrap aquery_llm so the sentinel is enqueued exactly once the
                    # query settles — on success, error, or cancellation. The
                    # finally runs after every progress event the callback awaited
                    # inside aquery_llm, so the sentinel is always the last queue
                    # item. put_nowait never blocks on the unbounded queue, which
                    # keeps the finally safe even while a CancelledError propagates.
                    try:
                        return await _query_with_skill_sources(
                            rag,
                            request.query,
                            param=param,
                            system_prompt=skill_system_prompt,
                            skill_plan=skill_plan,
                            progress_callback=progress_callback,
                        )
                    finally:
                        progress_queue.put_nowait(done_sentinel)

                query_task = asyncio.create_task(run_query())

                include_references = request.include_references
                include_chunk_content = request.include_chunk_content

                async def merged_generator():
                    try:
                        # Drain progress events until the sentinel: a blocking get
                        # with no polling, no timeout, and no dropped-item race.
                        while True:
                            event = await progress_queue.get()
                            if event is done_sentinel:
                                break
                            yield f"{json.dumps({'progress': event})}\n"

                        # Surface any exception from the task (e.g. LLM service
                        # error). Since the StreamingResponse has already begun
                        # (progress lines sent), we cannot raise an HTTP 500;
                        # instead emit a structured NDJSON error line so the
                        # client receives a well-formed error instead of a
                        # truncated stream.
                        try:
                            result = await query_task
                        except Exception as e:
                            logger.error(
                                f"Error in progress-enabled streaming query: {str(e)}",
                                exc_info=True,
                            )
                            yield f"{json.dumps({'error': str(e)})}\n"
                            return

                        # Yield references + LLM response chunks + response_time.
                        stream_gen = _build_stream_generator(
                            result=result,
                            include_references=include_references,
                            include_chunk_content=include_chunk_content,
                            include_response_time=True,
                            start_time=start_time,
                            skill_plan=_public_skill_plan(skill_plan),
                        )
                        async for line in stream_gen():
                            yield line
                    finally:
                        if not query_task.done():
                            query_task.cancel()
                            # Wait for cancellation cleanup so the task cannot
                            # outlive the response generator after disconnect.
                            await asyncio.gather(query_task, return_exceptions=True)

                return StreamingResponse(
                    merged_generator(),
                    media_type="application/x-ndjson",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "application/x-ndjson",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                # Default path: no progress events, original protocol order preserved.
                result = await _query_with_skill_sources(
                    rag,
                    request.query,
                    param=param,
                    system_prompt=skill_system_prompt,
                    skill_plan=skill_plan,
                )
                stream_gen = _build_stream_generator(
                    result=result,
                    include_references=request.include_references,
                    include_chunk_content=request.include_chunk_content,
                    include_response_time=False,
                    start_time=start_time,
                    skill_plan=_public_skill_plan(skill_plan),
                )

                return StreamingResponse(
                    stream_gen(),
                    media_type="application/x-ndjson",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "application/x-ndjson",
                        "X-Accel-Buffering": "no",
                    },
                )
        except Exception as e:
            logger.error(f"Error processing streaming query: {str(e)}", exc_info=True)
            raise internal_server_error(e)

    @router.post(
        "/query/data",
        response_model=QueryDataResponse,
        dependencies=[Depends(combined_auth)],
        responses={
            200: {
                "description": "Successful data retrieval response with structured RAG data",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {
                                    "type": "string",
                                    "enum": ["success", "failure"],
                                    "description": "Query execution status",
                                },
                                "message": {
                                    "type": "string",
                                    "description": "Status message describing the result",
                                },
                                "data": {
                                    "type": "object",
                                    "properties": {
                                        "entities": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "entity_name": {"type": "string"},
                                                    "entity_type": {"type": "string"},
                                                    "description": {"type": "string"},
                                                    "source_id": {"type": "string"},
                                                    "file_path": {"type": "string"},
                                                    "reference_id": {"type": "string"},
                                                },
                                            },
                                            "description": "Retrieved entities from knowledge graph",
                                        },
                                        "relationships": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "src_id": {"type": "string"},
                                                    "tgt_id": {"type": "string"},
                                                    "description": {"type": "string"},
                                                    "keywords": {"type": "string"},
                                                    "weight": {"type": "number"},
                                                    "source_id": {"type": "string"},
                                                    "file_path": {"type": "string"},
                                                    "reference_id": {"type": "string"},
                                                },
                                            },
                                            "description": "Retrieved relationships from knowledge graph",
                                        },
                                        "chunks": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "content": {"type": "string"},
                                                    "file_path": {"type": "string"},
                                                    "chunk_id": {"type": "string"},
                                                    "reference_id": {"type": "string"},
                                                },
                                            },
                                            "description": "Retrieved text chunks from vector database",
                                        },
                                        "references": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "reference_id": {"type": "string"},
                                                    "file_path": {"type": "string"},
                                                },
                                            },
                                            "description": "Reference list for citation purposes",
                                        },
                                    },
                                    "description": "Structured retrieval data containing entities, relationships, chunks, and references",
                                },
                                "metadata": {
                                    "type": "object",
                                    "properties": {
                                        "query_mode": {"type": "string"},
                                        "keywords": {
                                            "type": "object",
                                            "properties": {
                                                "high_level": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "low_level": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            },
                                        },
                                        "processing_info": {
                                            "type": "object",
                                            "properties": {
                                                "total_entities_found": {
                                                    "type": "integer"
                                                },
                                                "total_relations_found": {
                                                    "type": "integer"
                                                },
                                                "entities_after_truncation": {
                                                    "type": "integer"
                                                },
                                                "relations_after_truncation": {
                                                    "type": "integer"
                                                },
                                                "final_chunks_count": {
                                                    "type": "integer"
                                                },
                                            },
                                        },
                                    },
                                    "description": "Query metadata including mode, keywords, and processing information",
                                },
                            },
                            "required": ["status", "message", "data", "metadata"],
                        },
                        "examples": {
                            "successful_local_mode": {
                                "summary": "Local mode data retrieval",
                                "description": "Example of structured data from local mode query focusing on specific entities",
                                "value": {
                                    "status": "success",
                                    "message": "Query executed successfully",
                                    "data": {
                                        "entities": [
                                            {
                                                "entity_name": "Neural Networks",
                                                "entity_type": "CONCEPT",
                                                "description": "Computational models inspired by biological neural networks",
                                                "source_id": "chunk-123",
                                                "file_path": "/documents/ai_basics.pdf",
                                                "reference_id": "1",
                                            }
                                        ],
                                        "relationships": [
                                            {
                                                "src_id": "Neural Networks",
                                                "tgt_id": "Machine Learning",
                                                "description": "Neural networks are a subset of machine learning algorithms",
                                                "keywords": "subset, algorithm, learning",
                                                "weight": 1.0,
                                                "source_id": "chunk-123",
                                                "file_path": "/documents/ai_basics.pdf",
                                                "reference_id": "1",
                                            }
                                        ],
                                        "chunks": [
                                            {
                                                "content": "Neural networks are computational models that mimic the way biological neural networks work...",
                                                "file_path": "/documents/ai_basics.pdf",
                                                "chunk_id": "chunk-123",
                                                "reference_id": "1",
                                            }
                                        ],
                                        "references": [
                                            {
                                                "reference_id": "1",
                                                "file_path": "/documents/ai_basics.pdf",
                                            }
                                        ],
                                    },
                                    "metadata": {
                                        "query_mode": "local",
                                        "keywords": {
                                            "high_level": ["neural", "networks"],
                                            "low_level": [
                                                "computation",
                                                "model",
                                                "algorithm",
                                            ],
                                        },
                                        "processing_info": {
                                            "total_entities_found": 5,
                                            "total_relations_found": 3,
                                            "entities_after_truncation": 1,
                                            "relations_after_truncation": 1,
                                            "final_chunks_count": 1,
                                        },
                                    },
                                },
                            },
                            "global_mode": {
                                "summary": "Global mode data retrieval",
                                "description": "Example of structured data from global mode query analyzing broader patterns",
                                "value": {
                                    "status": "success",
                                    "message": "Query executed successfully",
                                    "data": {
                                        "entities": [],
                                        "relationships": [
                                            {
                                                "src_id": "Artificial Intelligence",
                                                "tgt_id": "Machine Learning",
                                                "description": "AI encompasses machine learning as a core component",
                                                "keywords": "encompasses, component, field",
                                                "weight": 2.0,
                                                "source_id": "chunk-456",
                                                "file_path": "/documents/ai_overview.pdf",
                                                "reference_id": "2",
                                            }
                                        ],
                                        "chunks": [],
                                        "references": [
                                            {
                                                "reference_id": "2",
                                                "file_path": "/documents/ai_overview.pdf",
                                            }
                                        ],
                                    },
                                    "metadata": {
                                        "query_mode": "global",
                                        "keywords": {
                                            "high_level": [
                                                "artificial",
                                                "intelligence",
                                                "overview",
                                            ],
                                            "low_level": [],
                                        },
                                    },
                                },
                            },
                            "naive_mode": {
                                "summary": "Naive mode data retrieval",
                                "description": "Example of structured data from naive mode using only vector search",
                                "value": {
                                    "status": "success",
                                    "message": "Query executed successfully",
                                    "data": {
                                        "entities": [],
                                        "relationships": [],
                                        "chunks": [
                                            {
                                                "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers...",
                                                "file_path": "/documents/deep_learning.pdf",
                                                "chunk_id": "chunk-789",
                                                "reference_id": "3",
                                            }
                                        ],
                                        "references": [
                                            {
                                                "reference_id": "3",
                                                "file_path": "/documents/deep_learning.pdf",
                                            }
                                        ],
                                    },
                                    "metadata": {
                                        "query_mode": "naive",
                                        "keywords": {"high_level": [], "low_level": []},
                                    },
                                },
                            },
                        },
                    }
                },
            },
            400: {
                "description": "Bad Request - Request validation failed",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                                "message": {"type": "string"},
                                "data": {"type": "object"},
                                "metadata": {"type": "object"},
                            },
                        },
                        "example": {
                            "status": "failure",
                            "message": (
                                "Validation error: body: Value error, RAG query is "
                                "too short. Enter at least 3 English characters or "
                                "an equivalent combination where each Chinese, "
                                "Japanese or Korean character counts as 2."
                            ),
                            "data": {},
                            "metadata": {},
                        },
                    }
                },
            },
            500: {
                "description": "Internal Server Error - Data retrieval failed",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"detail": {"type": "string"}},
                        },
                        "example": {
                            "detail": "Failed to retrieve data: Knowledge graph unavailable"
                        },
                    }
                },
            },
        },
    )
    async def query_data(request: QueryRequest):
        """
        Advanced data retrieval endpoint for structured RAG analysis.

        This endpoint provides raw retrieval results without LLM generation, perfect for:
        - **Data Analysis**: Examine what information would be used for RAG
        - **System Integration**: Get structured data for custom processing
        - **Debugging**: Understand retrieval behavior and quality
        - **Research**: Analyze knowledge graph structure and relationships

        **Key Features:**
        - No LLM generation - pure data retrieval
        - Complete structured output with entities, relationships, and chunks
        - Always includes references for citation
        - Detailed metadata about processing and keywords
        - Compatible with all query modes and parameters

        **Query Mode Behaviors:**
        - **local**: Returns entities and their direct relationships + related chunks
        - **global**: Returns relationship patterns across the knowledge graph
        - **hybrid**: Combines local and global retrieval strategies
        - **naive**: Returns only vector-retrieved text chunks (no knowledge graph)
        - **mix**: Integrates knowledge graph data with vector-retrieved chunks
        - **bypass**: Returns empty data arrays (used for direct LLM queries)

        **Data Structure:**
        - **entities**: Knowledge graph entities with descriptions and metadata
        - **relationships**: Connections between entities with weights and descriptions
        - **chunks**: Text segments from documents with source information
        - **references**: Citation information mapping reference IDs to file paths
        - **metadata**: Processing information, keywords, and query statistics

        **Usage Examples:**

        Analyze entity relationships:
        ```json
        {
            "query": "machine learning algorithms",
            "mode": "local",
            "top_k": 10
        }
        ```

        Explore global patterns:
        ```json
        {
            "query": "artificial intelligence trends",
            "mode": "global",
            "max_relation_tokens": 2000
        }
        ```

        Vector similarity search:
        ```json
        {
            "query": "neural network architectures",
            "mode": "naive",
            "chunk_top_k": 5
        }
        ```

        Bypass initial LLM call by providing high-level and low-level keywords:
        ```json
        {
            "query": "What is Retrieval-Augmented-Generation?",
            "hl_keywords": ["machine learning", "information retrieval", "natural language processing"],
            "ll_keywords": ["retrieval augmented generation", "RAG", "knowledge base"],
            "mode": "mix"
        }
        ```

        **Response Analysis:**
        - **Empty arrays**: Normal for certain modes (e.g., naive mode has no entities/relationships)
        - **Processing info**: Shows retrieval statistics and token usage
        - **Keywords**: High-level and low-level keywords extracted from query
        - **Reference mapping**: Links all data back to source documents

        Args:
            request (QueryRequest): The request object containing query parameters:
                - **query**: The search query to analyze. Must not be empty. RAG modes additionally require an English-equivalent length of at least 3; each Chinese, Japanese or Korean character counts as 2. Bypass mode is exempt from the minimum, not from being non-empty.
                - **mode**: Retrieval strategy affecting data types returned
                - **top_k**: Number of top entities/relationships to retrieve
                - **chunk_top_k**: Number of text chunks to retrieve
                - **max_entity_tokens**: Token limit for entity context
                - **max_relation_tokens**: Token limit for relationship context
                - **max_total_tokens**: Overall token budget for retrieval

        Returns:
            QueryDataResponse: Structured JSON response containing:
                - **status**: "success" or "failure"
                - **message**: Human-readable status description
                - **data**: Complete retrieval results with entities, relationships, chunks, references
                - **metadata**: Query processing information and statistics

        Raises:
            HTTPException:
                - 400: Request validation failed (e.g., query empty or too short, invalid mode)
                - 500: Internal processing error (e.g., knowledge graph unavailable)

        Note:
            This endpoint always includes references regardless of the include_references parameter,
            as structured data analysis typically requires source attribution.
        """
        try:
            param = request.to_query_params(False)  # No streaming for data endpoint
            response = await rag.aquery_data(request.query, param=param)

            # aquery_data returns the new format with status, message, data, and metadata
            if isinstance(response, dict):
                return QueryDataResponse(**response)
            else:
                # Handle unexpected response format
                return QueryDataResponse(
                    status="failure",
                    message="Invalid response type",
                    data={},
                    metadata={},
                )
        except Exception as e:
            logger.error(f"Error processing data query: {str(e)}", exc_info=True)
            raise internal_server_error(e)

    from eve_lightrag.eve_esi_routes import create_esi_routes
    router.include_router(create_esi_routes(
        get_combined_auth_dependency(api_key, respect_whitelist=False)
    ))
    return router
