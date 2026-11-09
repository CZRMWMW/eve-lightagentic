"""Query skills for LightRAG's graph, vector, and EVE data sources."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import os
import re
from typing import Any


SKILL_NAMES = (
    "neo4j_knowledge_graph",
    "milvus_vector_search",
    "eve_fitting_mysql",
    "eve_static_data_mysql",
    "pyfa_fitting_simulator",
    "eve_character_skills_esi",
)

SKILL_DESCRIPTIONS = {
    "neo4j_knowledge_graph": "查询 Neo4j 知识图谱中的实体、上下位关系和关联事实。",
    "milvus_vector_search": "从 Milvus 本地文档库中进行语义检索，查找原文、说明和证据。",
    "eve_fitting_mysql": "查询 MySQL 中的 EVE 舰船装配方案和槽位配置。",
    "eve_static_data_mysql": "查询 MySQL 中的 EVE SDE 物品、装备、舰船、蓝图、星系和属性数据。",
    "pyfa_fitting_simulator": "调用本地 Pyfa/Eos 计算引擎，验证装配并计算 CPU、能量栅格、槽位、EHP、电容和 DPS。",
    "eve_character_skills_esi": "通过已授权角色的 ESI 读取当前技能等级、技能点和训练队列；不返回任何令牌。",
}

DEFAULT_GAME_CONTEXT = (
    "默认游戏语境：用户没有明确指定其他游戏时，所有游戏术语、舰船、刷怪、"
    "配置、技能、收益和玩法问题均按《EVE Online》解释。"
    "例如“毒蜥”默认指 EVE Online 的毒蜥级（Gila），"
    "“配置”默认指舰船装配（fit/loadout），不是电脑硬件配置。"
    "只有用户明确提到其他游戏时，才切换到其他游戏语境。"
    "EVE术语可能是舰船简称、物品名或玩家黑话，不能只按字面含义扩写。"
    "涉及游戏机制、数值、操作步骤、舰船定位的事实必须有本次查询结果支持；"
    "检索命中不等于支持结论，不能把仅有关键词关联的段落当作定义。"
    "存在歧义时先说明资料中能确认的含义，再询问用户语境。"
    "资料未覆盖的机制应明确说无法确认，不得编造税费、技能、装备或操作。"
)

_EXECUTORS: dict[str, ThreadPoolExecutor] = {}


def _env_int(name: str, default: int, minimum: int = 1, maximum: int = 128) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _get_executor(name: str, max_workers: int) -> ThreadPoolExecutor:
    executor = _EXECUTORS.get(name)
    if executor is None:
        executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"eve-{name}",
        )
        _EXECUTORS[name] = executor
    return executor


def _get_loop_semaphore(name: str, capacity: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    attr = f"_eve_skill_semaphore_{name}"
    semaphore = getattr(loop, attr, None)
    if semaphore is None:
        semaphore = asyncio.Semaphore(capacity)
        setattr(loop, attr, semaphore)
    return semaphore


async def run_eve_db_call(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run blocking EVE MySQL work on a dedicated bounded executor."""
    max_workers = _env_int("EVE_DB_THREAD_WORKERS", 8, 1, 64)
    concurrency = _env_int("EVE_DB_MAX_CONCURRENCY", max_workers, 1, 64)
    executor = _get_executor("db", max_workers)
    semaphore = _get_loop_semaphore("db", concurrency)
    call = partial(func, *args, **kwargs)
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, call)








def _parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}




async def extract_fitting_targets(
    rag: Any,
    query: str,
    research_results: list[dict[str, Any]],
    conversation_history: list[dict[str, Any]] | None = None,
    max_targets: int = 6,
) -> dict[str, Any]:
    """Extract concrete ship names only after graph/vector research."""
    research_text = format_skill_context(research_results)
    if len(research_text) > 18000:
        research_text = research_text[:18000] + "\n[研究结果已截断]"
    system_prompt = (
        "你是 EVE 装配查询的中间 Agent。"
        + DEFAULT_GAME_CONTEXT
        + "\n"
        "你不回答用户问题，只从用户问题和前置检索结果中提取适合继续查装配的具体舰船名。"
        "只输出 JSON，不要 Markdown，不要输出思考过程，reason 必须使用中文。\n"
        "要求：\n"
        "1. 只提取明确的舰船名称或可以直接用于装配库搜索的名称。\n"
        "2. 不要把玩法、装备、NPC、地区或技能名当成舰船。\n"
        f"3. 最多返回 {max_targets} 艘舰船；没有可靠舰船时返回空数组。\n"
        '格式：{"targets":[{"query":"舰船名","reason":"来源或用途"}]}'
    )
    prompt = (
        f"用户问题：\n{query}\n\n"
        f"前置检索结果：\n{research_text}"
    )
    global_config = rag._build_global_config()
    llm = global_config["role_llm_funcs"]["query"]
    try:
        raw = await llm(
            prompt,
            system_prompt=system_prompt,
            history_messages=conversation_history or [],
            enable_cot=False,
            stream=False,
        )
        parsed = _parse_json_object(raw if isinstance(raw, str) else str(raw))
        targets = parsed.get("targets", [])
        if not isinstance(targets, list):
            targets = []
        cleaned: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in targets:
            if isinstance(item, str):
                candidate = item.strip()
                reason = ""
            elif isinstance(item, dict):
                candidate = str(item.get("query", "")).strip()
                reason = str(item.get("reason", "")).strip()
            else:
                continue
            if candidate and candidate.casefold() not in seen:
                cleaned.append({"query": candidate, "reason": reason})
                seen.add(candidate.casefold())
            if len(cleaned) >= max_targets:
                break
        return {
            "ok": True,
            "targets": cleaned,
            "raw": raw if isinstance(raw, str) else str(raw),
        }
    except Exception as exc:
        return {"ok": False, "targets": [], "reason": str(exc), "raw": ""}


async def extract_simulation_payload(
    rag: Any,
    query: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract an explicit ship/module payload for the Pyfa simulator."""
    from .eve_names import name_context
    glossary = await run_eve_db_call(name_context, query)
    system_prompt = (
        "你是 EVE 装配模拟请求解析器。"
        + DEFAULT_GAME_CONTEXT
        + "\n只输出 JSON，不要 Markdown、解释或思考过程。"
        "从用户问题中提取明确写出的舰船和装备；不要凭记忆补充没有出现的装备。"
        "如果用户没有提供装备，槽位数组返回空数组，仍可模拟舰船基础属性。"
        "槽位只能使用 high、mid、low、rig、subsystem、service。"
        "技能字段：skills_source 可为 default、all5、all0、esi；"
        "真实角色可用 character_id 或 character 填写；skills 是技能名/typeID 到 0-5 的对象。"
        '格式：{"ship":"舰船名","skills_source":"default","character_id":null,"character":null,"skills":{},"high":[],"mid":[],"low":[],"rig":[],"subsystem":[],"service":[]}'
    )
    global_config = rag._build_global_config()
    llm = global_config["role_llm_funcs"]["query"]
    try:
        raw = await llm(
            f"用户问题：\n{query.strip()}" + glossary,
            system_prompt=system_prompt,
            history_messages=conversation_history or [],
            enable_cot=False,
            stream=False,
        )
        parsed = _parse_json_object(raw if isinstance(raw, str) else str(raw))
        if not isinstance(parsed.get("ship"), (str, int)):
            return {"ok": False, "payload": {}, "raw": str(raw)}
        payload: dict[str, Any] = {"ship": parsed["ship"]}
        for key in ("skills_source", "character_id", "character"):
            if parsed.get(key) is not None:
                payload[key] = parsed[key]
        payload["skills"] = parsed.get("skills", {}) if isinstance(parsed.get("skills", {}), dict) else {}
        normalized_query = query.casefold()
        if re.search(
            r"(我的角色|按我的技能|按角色技能|真实技能|角色技能|esi技能)",
            normalized_query,
            re.IGNORECASE,
        ):
            payload["skills_source"] = "esi"
        elif re.search(r"(全五|全5|all\s*5|技能\s*5级)", normalized_query, re.IGNORECASE):
            payload["skills_source"] = "all5"
        elif re.search(r"(全零|全0|all\s*0|技能\s*0级)", normalized_query, re.IGNORECASE):
            payload["skills_source"] = "all0"
        for slot in ("high", "mid", "low", "rig", "subsystem", "service"):
            value = parsed.get(slot, [])
            payload[slot] = value if isinstance(value, list) else []
        return {
            "ok": True,
            "payload": payload,
            "raw": raw if isinstance(raw, str) else str(raw),
        }
    except Exception as exc:
        return {"ok": False, "payload": {}, "reason": str(exc), "raw": ""}




async def query_neo4j(
    rag: Any,
    query: str,
    limit: int = 20,
    max_depth: int = 1,
) -> dict[str, Any]:
    """Search Neo4j labels and return their nodes and immediate relationships."""
    graph = rag.chunk_entity_relation_graph
    labels = await graph.search_labels(query, limit=max(1, min(limit, 50)))
    labels = labels[: max(1, min(limit, 50))]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()
    for label in labels:
        node = await graph.get_node(label)
        if node:
            nodes.append({"id": label, "properties": dict(node)})
            seen_nodes.add(label)
        pairs = await graph.get_node_edges(label) or []
        for source, target in pairs[: max(1, limit)]:
            edge_key = (str(source), str(target))
            if edge_key in seen_edges:
                continue
            edge = await graph.get_edge(source, target)
            if edge:
                edges.append(
                    {
                        "source": str(source),
                        "target": str(target),
                        "properties": dict(edge),
                    }
                )
                seen_edges.add(edge_key)
            for endpoint in (str(source), str(target)):
                if endpoint not in seen_nodes and len(nodes) < limit * 2:
                    endpoint_node = await graph.get_node(endpoint)
                    if endpoint_node:
                        nodes.append(
                            {
                                "id": endpoint,
                                "properties": dict(endpoint_node),
                            }
                        )
                        seen_nodes.add(endpoint)
    return {
        "skill": "neo4j_knowledge_graph",
        "query": query,
        "labels": labels,
        "nodes": nodes[: limit * 2],
        "edges": edges[: limit * 2],
        "max_depth": max_depth,
    }


async def query_milvus(
    rag: Any,
    query: str,
    limit: int = 20,
) -> dict[str, Any]:
    """Run a direct semantic search against the LightRAG Milvus chunk store."""
    rows = await rag.chunks_vdb.query(query, max(1, min(limit, 100)))
    return {
        "skill": "milvus_vector_search",
        "query": query,
        "results": rows[: max(1, min(limit, 100))],
    }




def format_skill_context(results: list[dict[str, Any]]) -> str:
    """Format skill results as authoritative context for the answer LLM."""
    sections = [
        "以下是由 LightRAG 查询技能返回的本地数据。",
        "只能依据已返回的技能结果回答；结果没有覆盖的内容不要编造。",
    ]
    for result in results:
        skill = result.get("skill", "unknown")
        sections.append(
            f"\n【技能结果：{skill}】\n"
            + json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        )
    return "\n".join(sections)
