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


def needs_multi_layer_agent(query: str) -> bool:
    """Return whether the query needs research first and fitting lookup second."""
    normalized = " ".join((query or "").casefold().split())
    has_income = bool(
        re.search(
            r"(isk|赚钱|赚取|收益|收入|刷钱|挣钱|致富|怎么赚|如何赚)",
            normalized,
            re.IGNORECASE,
        )
    )
    has_ship_or_fitting = bool(
        re.search(
            r"(什么船|哪条船|哪些船|用船|舰船|飞船|配装|装配|配船|怎么配|如何配|fit(?:ting)?|loadout)",
            normalized,
            re.IGNORECASE,
        )
    )
    return has_income and has_ship_or_fitting


def needs_skill_fitting_recommendation(query: str) -> bool:
    """Return whether the user wants a fit selected against real character skills."""
    normalized = " ".join((query or "").casefold().split())
    has_character_context = bool(
        re.search(
            r"(我的技能|我现在的技能|按我的技能|按角色技能|真实技能|"
            r"角色技能|esi技能|适合我|适配我|技能适配|技能匹配|"
            r"当前技能|现有技能|技能情况|技能底子|"
            r"角色\s*(?:id|编号|名|名字))",
            normalized,
            re.IGNORECASE,
        )
    )
    has_recommendation_intent = bool(
        re.search(
            r"(推荐|建议|定制|适合|合适|匹配|选择|挑一套|哪一套|"
            r"最适合|能用的|可用的).{0,20}"
            r"(配置|装配|配装|配船|fit|loadout|舰船)",
            normalized,
            re.IGNORECASE,
        )
        or re.search(
            r"(配置|装配|配装|配船|fit|loadout|舰船).{0,20}"
            r"(推荐|建议|定制|适合|合适|匹配|哪一套|能用)",
            normalized,
            re.IGNORECASE,
        )
    )
    return has_character_context and has_recommendation_intent


def needs_character_fitting_analysis(query: str) -> bool:
    """Return whether real character skills must be used to calculate fit metrics."""
    normalized = " ".join((query or "").casefold().split())
    has_character_context = bool(
        re.search(
            r"(我的技能|我现在的技能|按我的技能|按角色技能|真实技能|角色技能|"
            r"esi技能|当前技能|现有技能|技能情况|"
            r"角色\s*(?:id|编号|名|名字))",
            normalized,
            re.IGNORECASE,
        )
    )
    has_metric = bool(
        re.search(
            r"(cpu|pg|电网|能量格栅|能量栅格|回盾|护盾回充|盾修|"
            r"电容|cap|dps|伤害|输出|ehp|有效生命|槽位)",
            normalized,
            re.IGNORECASE,
        )
    )
    has_fit_reference = bool(
        re.search(
            r"(装配|配装|配船|配置|方案|这套|pve|刷怪|速刷|"
            r"盾导弹|导弹流|舰船|fit|loadout|pyfa)",
            normalized,
            re.IGNORECASE,
        )
    )
    return has_character_context and has_metric and has_fit_reference


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


async def plan_skills(
    rag: Any,
    query: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ask the configured query LLM which data skills are needed."""
    system_prompt = (
        "你是 LightRAG 的查询技能路由器。你只能从给定技能中选择需要调用的技能，"
        "不要回答用户问题，不要输出分析过程。必须只输出 JSON，不要 Markdown。\n"
        + DEFAULT_GAME_CONTEXT
        + "\n"
        "技能说明：\n"
        + "\n".join(f"- {name}: {desc}" for name, desc in SKILL_DESCRIPTIONS.items())
        + "\n规则：\n"
        "1. 纯闲聊、创作、翻译或不需要本地资料的问题，skills 返回空数组 []。\n"
        "2. 需要知识库实体关系、上下位关系、关联关系时选择 neo4j_knowledge_graph。\n"
        "3. 需要从本地文档按语义查找原文、说明、段落或证据时选择 milvus_vector_search。\n"
        "4. EVE 舰船装配只选择 eve_fitting_mysql，不要选择 eve_static_data_mysql。\n"
        "5. EVE 装备、舰船、蓝图、星系、物品属性和 SDE 信息选择 eve_static_data_mysql。\n"
        "6. 用户明确要求模拟、验证装配，或询问 CPU/能量栅格/槽位是否够、DPS、EHP、抗性、电容时选择 pyfa_fitting_simulator；"
        "这类问题通常还需要 eve_static_data_mysql。\n"
        "7. 用户询问我的角色技能、我的角色能否使用/装配、按真实角色技能计算，选择 eve_character_skills_esi；"
        "仅“全五模拟/全零模拟/指定技能等级模拟”不需要 ESI。\n"
        "8. 一个问题可以同时选择多个技能，按照实际依赖选择，不要为了凑数量调用技能。\n"
        "9. 用户使用中文时，reason 必须用中文填写。\n"
        '输出格式：{"skills":["技能名"],"reason":"简短中文原因"}'
    )
    from eve_lightrag.eve_names import name_context
    prompt = f"用户问题：\n{query.strip()}" + await run_eve_db_call(name_context, query)
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
        selected = parsed.get("skills", [])
        if not isinstance(selected, list):
            selected = []
        selected = [skill for skill in selected if skill in SKILL_NAMES]
        normalized_query = query.casefold()
        # ISK-making and beginner gameplay questions need both the structured
        # relationship context and the source-document explanations.
        if (
            re.search(r"isk", normalized_query)
            and re.search(r"(新人|新手|赚钱|赚取|收益|收入|刷钱|挣钱)", normalized_query)
        ):
            selected = ["neo4j_knowledge_graph", "milvus_vector_search"]
            parsed["reason"] = (
                "新人ISK收益问题同时需要知识图谱中的玩法关联和文档中的具体指南"
            )
        if needs_multi_layer_agent(query):
            selected = [
                "neo4j_knowledge_graph",
                "milvus_vector_search",
                "eve_fitting_mysql",
            ]
            parsed["reason"] = (
                "这是赚钱路线、推荐舰船和舰船装配的复合问题，需要分层查询"
            )
        if needs_skill_fitting_recommendation(query):
            selected = [
                "eve_character_skills_esi",
                "eve_fitting_mysql",
                "pyfa_fitting_simulator",
            ]
            parsed["reason"] = (
                "用户要求按已授权角色的真实技能，从装配库候选方案中调用 Pyfa "
                "逐套验证并推荐可用配置"
            )
        if needs_character_fitting_analysis(query):
            selected = [
                "eve_character_skills_esi",
                "eve_fitting_mysql",
                "pyfa_fitting_simulator",
            ]
            parsed["reason"] = (
                "用户要求按角色真实技能对装配进行 Pyfa 计算，返回 CPU、"
                "能量格栅、护盾回充、电容和 DPS 等指标"
            )
        if re.search(
            r"(pyfa|模拟|验证|查看属性|计算属性|丢到|扔到).{0,20}"
            r"(装配|配置|配装|fit|舰船|洛基|金鹏|tengu|loki)"
            r"|"
            r"(装配|配置|配装|fit|舰船|洛基|金鹏|tengu|loki).{0,20}"
            r"(pyfa|模拟|验证|查看属性|计算属性|丢到|扔到)",
            normalized_query,
            re.IGNORECASE,
        ):
            if "eve_fitting_mysql" not in selected:
                selected.append("eve_fitting_mysql")
            if "pyfa_fitting_simulator" not in selected:
                selected.append("pyfa_fitting_simulator")
            parsed["reason"] = (
                "用户要求从装配库选择装配并调用 Pyfa 计算属性"
            )
        if re.search(
            r"(网子|停滞缠绕|停滞缠绕光束|stasis\s+webifier)",
            normalized_query,
            re.IGNORECASE,
        ) and re.search(
            r"(类型|型号|分类|变体|有哪些)",
            normalized_query,
            re.IGNORECASE,
        ):
            selected = ["eve_static_data_mysql"]
            parsed["reason"] = (
                "这是停滞缠绕光束类型目录查询，只读取 EVE SDE 静态数据中的实际模块"
            )
        if re.search(
            r"(我的角色|按我的技能|按角色技能|真实技能|角色技能|esi技能|技能方案|"
            r"(?:使用|用).{0,20}(?:技能|技能方案)|"
            r"角色\s*(?:id|编号|名|名字)|"
            r"(?:读取|查询).{0,20}(?:用户|角色).{0,10}技能)",
            normalized_query,
            re.IGNORECASE,
        ):
            if "eve_character_skills_esi" not in selected:
                selected.append("eve_character_skills_esi")
            if re.search(
                r"(能不能|是否能|可以装|装得上|模拟|验证|cpu|能量栅格|"
                r"dps|ehp|抗性|电容|缺失技能|技能点|技能等级|sp)",
                normalized_query,
                re.IGNORECASE,
            ) and "pyfa_fitting_simulator" not in selected:
                selected.append("pyfa_fitting_simulator")
            parsed["reason"] = (
                "用户要求读取或使用角色真实技能，必须调用 ESI 角色技能 skill"
            )
        return {
            "ok": True,
            "skills": list(dict.fromkeys(selected)),
            "reason": str(parsed.get("reason", "")).strip(),
            "raw": raw if isinstance(raw, str) else str(raw),
        }
    except Exception as exc:
        return {
            "ok": False,
            "skills": [],
            "reason": f"skill planner failed: {exc}",
            "raw": "",
        }


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
    from eve_lightrag.eve_names import name_context
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


async def run_multi_layer_agent(
    rag: Any,
    query: str,
    conversation_history: list[dict[str, Any]] | None,
    fitting_lookup: Any,
    fitting_formatter: Any,
    fitting_target_discoverer: Any | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Run research first, then dependent ship-fitting lookups."""
    research_tasks = [
        query_neo4j(rag, query, limit),
        query_milvus(rag, query, limit),
    ]
    raw_research = await asyncio.gather(
        *research_tasks, return_exceptions=True
    )
    research_results: list[dict[str, Any]] = []
    for result in raw_research:
        if isinstance(result, Exception):
            research_results.append(
                {"skill": "query_skill_error", "error": str(result)}
            )
        else:
            research_results.append(result)

    target_plan = await extract_fitting_targets(
        rag,
        query,
        research_results,
        conversation_history,
    )
    if not target_plan.get("targets") and fitting_target_discoverer is not None:
        research_text = format_skill_context(research_results)
        try:
            fallback_targets = await run_eve_db_call(
                fitting_target_discoverer,
                query + "\n" + research_text,
                6,
            )
        except Exception as exc:
            fallback_targets = []
            target_plan["fallback_error"] = str(exc)
        if fallback_targets:
            target_plan["targets"] = fallback_targets
            target_plan["fallback"] = True
    fitting_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    targets = target_plan.get("targets", [])

    async def _lookup_target(
        target: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        try:
            rows = await run_eve_db_call(
                fitting_lookup,
                target["query"],
                None,
            )
        except Exception as exc:
            rows = []
            target["error"] = str(exc)
        return target, rows or []

    lookup_results = await asyncio.gather(
        *[_lookup_target(target) for target in targets],
        return_exceptions=True,
    )
    for result in lookup_results:
        if isinstance(result, Exception):
            continue
        _target, rows = result
        for row in rows or []:
            row_id = str(row.get("id", ""))
            if row_id and row_id in seen_ids:
                continue
            if row_id:
                seen_ids.add(row_id)
            fitting_rows.append(row)

    return {
        "skill": "eve_income_fitting_multi_layer",
        "query": query,
        "stages": [
            "research_neo4j_and_milvus",
            "extract_ship_targets",
            "query_mysql_fittings",
        ],
        "research": research_results,
        "ship_targets": target_plan.get("targets", []),
        "fittings": fitting_rows,
        "fitting_context": fitting_formatter(fitting_rows) if fitting_rows else "",
        "context": (
            format_skill_context(research_results)
            + (
                "\n\n【依赖查询：舰船装配 MySQL 结果】\n"
                + fitting_formatter(fitting_rows)
                if fitting_rows
                else "\n\n【依赖查询：舰船装配 MySQL 结果】\n没有找到候选舰船的装配方案。"
            )
        ),
    }


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
