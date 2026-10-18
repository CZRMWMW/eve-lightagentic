"""MySQL-backed EVE fitting lookup for the LightRAG query API."""

from __future__ import annotations

import html
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree as ET

import pymysql
from pymysql.cursors import DictCursor


DEFAULT_ENV_FILE = Path(os.getenv("EVE_ENV_FILE", ".env"))
DEFAULT_SOURCE_DIR = Path(os.getenv("EVE_FITTINGS_SOURCE_DIR", "data/fittings"))
DEFAULT_DATABASE = "EVE"

_FITTING_INTENT_PATTERNS = (
    # Chinese fitting/loadout intent.
    r"装配",
    r"配船",
    r"配装",
    r"舰船配置",
    r"装备配置",
    r"装备清单",
    r"装配方案",
    r"装配推荐",
    r"模块搭配",
    r"配置问题",
    r"配置(?:怎么样|如何|怎么)",
    r"怎么配",
    r"如何配",
    r"推荐.{0,12}(装备|装配|配船|配装)",
    r"(装|带).{0,8}装备",
    r"(高槽|中槽|低槽|改装件|无人机舱|货舱|槽位)",
    r"(舰船|飞船).{0,12}(装备|模块|配置)",
    # English fitting/loadout intent.
    r"\bfit(?:ting)?\b",
    r"\bloadout\b",
    r"\bsetup\b",
    r"\bpyfa\b",
    r"\beft\b",
    r"\bhigh\s*slot\b",
    r"\bmid\s*slot\b",
    r"\blow\s*slot\b",
    r"\brig(?:s)?\b",
    r"\bdrone\s*bay\b",
)

_NON_FITTING_INTENT_PATTERNS = (
    r"(?:不要|不需要|无需|不用|不是|先不).{0,12}"
    r"(?:装配|配船|配装|装备|fit(?:ting)?|loadout)",
    r"(?:电脑|硬件|主机|显卡|系统).{0,8}配置",
)


def is_fitting_query(query: str) -> bool:
    """Return whether a user request is asking for a ship fitting/loadout."""
    if not query or not query.strip():
        return False
    normalized = " ".join(query.casefold().split())
    if any(
        re.search(pattern, normalized, re.IGNORECASE)
        for pattern in _NON_FITTING_INTENT_PATTERNS
    ):
        return False
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in _FITTING_INTENT_PATTERNS)


def fitting_lookup_query(query: str, conversation_history: list[dict[str, Any]] | None = None) -> str:
    """Include recent user turns so fitting follow-ups can resolve the ship name."""
    parts = [query.strip()]
    if conversation_history:
        for message in conversation_history[-6:]:
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                content = message["content"].strip()
                if content and content != query.strip():
                    parts.append(content)
    return "\n".join(parts)[-4000:]


def _load_dotenv(path: Path = DEFAULT_ENV_FILE) -> None:
    """Load only missing values so PM2/process environment keeps precedence."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _db_config() -> dict[str, Any]:
    return {
        "host": os.getenv("EVE_FITTINGS_DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("EVE_FITTINGS_DB_PORT", "3306")),
        "user": os.getenv("EVE_FITTINGS_DB_USER", "eve"),
        "password": os.getenv("EVE_FITTINGS_DB_PASSWORD", ""),
        "database": os.getenv("EVE_FITTINGS_DB_NAME", DEFAULT_DATABASE),
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
        "autocommit": False,
        "connect_timeout": int(os.getenv("EVE_FITTINGS_DB_CONNECT_TIMEOUT", "5")),
    }


def _clean_localized(value: str | None) -> tuple[str, str]:
    """Return (English name, localized name) from EVE's localized XML value."""
    if not value:
        return "", ""
    decoded = html.unescape(value).strip()
    match = re.search(
        r"<localized\s+hint=\"(?P<en>.*?)\">(?P<local>.*?)</localized>",
        decoded,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return "", decoded.rstrip("*").strip()
    english = html.unescape(match.group("en")).strip()
    localized = html.unescape(match.group("local")).rstrip("*").strip()
    return english, localized


def _iter_fittings(path: Path) -> Iterator[tuple[ET.Element, int]]:
    """Yield complete fitting elements, tolerating a truncated XML tail."""
    data = path.read_text(encoding="utf-8", errors="replace")
    starts = [m.start() for m in re.finditer(r"<fitting\b", data)]
    for index, start in enumerate(starts, 1):
        end = data.find("</fitting>", start)
        if end < 0:
            break
        fragment = data[start : end + len("</fitting>")]
        try:
            yield ET.fromstring(fragment), index
        except ET.ParseError:
            continue


def parse_source(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fitting, source_index in _iter_fittings(path):
        ship = fitting.find("shipType")
        ship_en, ship_local = _clean_localized(
            ship.attrib.get("value") if ship is not None else ""
        )
        description_node = fitting.find("description")
        description = (
            html.unescape(description_node.attrib.get("value", ""))
            if description_node is not None
            else ""
        )
        items: list[dict[str, Any]] = []
        for item_index, hardware in enumerate(fitting.findall("hardware")):
            item_en, item_local = _clean_localized(hardware.attrib.get("type"))
            slot = hardware.attrib.get("slot", "").strip()
            quantity = int(hardware.attrib.get("qty", "1") or "1")
            items.append(
                {
                    "item_index": item_index,
                    "slot_name": slot,
                    "quantity": quantity,
                    "item_name_en": item_en,
                    "item_name_local": item_local,
                    "raw_type": hardware.attrib.get("type", ""),
                }
            )
        search_text = " ".join(
            [
                fitting.attrib.get("name", ""),
                ship_en,
                ship_local,
                description,
                *(item["item_name_en"] for item in items),
                *(item["item_name_local"] for item in items),
            ]
        )
        records.append(
            {
                "source_file": path.name,
                "source_index": source_index,
                "fitting_name": fitting.attrib.get("name", "").strip(),
                "ship_name_en": ship_en,
                "ship_name_local": ship_local,
                "description": description,
                "search_text": search_text,
                "items": items,
            }
        )
    return records


def _identifiers() -> tuple[str, str]:
    return (
        os.getenv("EVE_FITTINGS_TABLE", "eve_fittings"),
        os.getenv("EVE_FITTING_ITEMS_TABLE", "eve_fitting_items"),
    )


@contextmanager
def connection() -> Iterator[pymysql.connections.Connection]:
    conn = pymysql.connect(**_db_config())
    try:
        yield conn
    finally:
        conn.close()


def ensure_schema(conn: pymysql.connections.Connection) -> None:
    fittings_table, items_table = _identifiers()
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS `{fittings_table}` (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                source_file VARCHAR(255) NOT NULL,
                source_index INT UNSIGNED NOT NULL,
                fitting_name VARCHAR(255) NOT NULL,
                ship_name_en VARCHAR(160) NOT NULL DEFAULT '',
                ship_name_local VARCHAR(160) NOT NULL DEFAULT '',
                description TEXT NULL,
                search_text TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (id),
                UNIQUE KEY uq_eve_fitting_source (source_file, source_index),
                KEY idx_eve_fitting_ship_en (ship_name_en),
                KEY idx_eve_fitting_ship_local (ship_name_local),
                KEY idx_eve_fitting_name (fitting_name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS `{items_table}` (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                fitting_id BIGINT UNSIGNED NOT NULL,
                item_index INT UNSIGNED NOT NULL,
                slot_name VARCHAR(64) NOT NULL DEFAULT '',
                quantity INT UNSIGNED NOT NULL DEFAULT 1,
                item_name_en VARCHAR(255) NOT NULL DEFAULT '',
                item_name_local VARCHAR(255) NOT NULL DEFAULT '',
                raw_type TEXT NULL,
                PRIMARY KEY (id),
                UNIQUE KEY uq_eve_fitting_item (fitting_id, item_index),
                KEY idx_eve_item_fitting (fitting_id),
                CONSTRAINT fk_eve_item_fitting
                    FOREIGN KEY (fitting_id) REFERENCES `{fittings_table}` (id)
                    ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )


def import_sources(source_dir: Path = DEFAULT_SOURCE_DIR) -> dict[str, int]:
    source_paths = sorted(source_dir.glob("*.xml"))
    parsed = [(path, parse_source(path)) for path in source_paths if path.is_file()]
    fittings_table, items_table = _identifiers()
    imported_fittings = 0
    imported_items = 0

    with connection() as conn:
        ensure_schema(conn)
        with conn.cursor() as cursor:
            for path, records in parsed:
                for record in records:
                    cursor.execute(
                        f"""
                        INSERT INTO `{fittings_table}`
                            (source_file, source_index, fitting_name,
                             ship_name_en, ship_name_local, description, search_text)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            fitting_name = VALUES(fitting_name),
                            ship_name_en = VALUES(ship_name_en),
                            ship_name_local = VALUES(ship_name_local),
                            description = VALUES(description),
                            search_text = VALUES(search_text),
                            id = LAST_INSERT_ID(id)
                        """,
                        (
                            record["source_file"],
                            record["source_index"],
                            record["fitting_name"],
                            record["ship_name_en"],
                            record["ship_name_local"],
                            record["description"],
                            record["search_text"],
                        ),
                    )
                    fitting_id = cursor.lastrowid
                    cursor.execute(
                        f"DELETE FROM `{items_table}` WHERE fitting_id = %s",
                        (fitting_id,),
                    )
                    for item in record["items"]:
                        cursor.execute(
                            f"""
                            INSERT INTO `{items_table}`
                                (fitting_id, item_index, slot_name, quantity,
                                 item_name_en, item_name_local, raw_type)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                fitting_id,
                                item["item_index"],
                                item["slot_name"],
                                item["quantity"],
                                item["item_name_en"],
                                item["item_name_local"],
                                item["raw_type"],
                            ),
                        )
                        imported_items += 1
                    imported_fittings += 1
        conn.commit()
    return {
        "source_files": len(parsed),
        "fittings": imported_fittings,
        "items": imported_items,
    }


def find_fittings(query: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Find fittings with bidirectional name matching for names and aliases."""
    fittings_table, items_table = _identifiers()
    if not query.strip():
        return []
    with connection() as conn, conn.cursor() as cursor:
        sql = f"""
            SELECT id, source_file, fitting_name, ship_name_en, ship_name_local,
                   description
            FROM `{fittings_table}`
            WHERE (
                (ship_name_en <> '' AND (
                    LOCATE(LOWER(ship_name_en), LOWER(%s)) > 0
                    OR LOCATE(LOWER(%s), LOWER(ship_name_en)) > 0
                ))
                OR (ship_name_local <> '' AND (
                    LOCATE(LOWER(ship_name_local), LOWER(%s)) > 0
                    OR LOCATE(LOWER(%s), LOWER(ship_name_local)) > 0
                    OR LOCATE(
                        LOWER(TRIM(TRAILING '级' FROM ship_name_local)),
                        LOWER(%s)
                    ) > 0
                ))
                OR (fitting_name <> '' AND (
                    LOCATE(LOWER(fitting_name), LOWER(%s)) > 0
                    OR LOCATE(LOWER(%s), LOWER(fitting_name)) > 0
                ))
            )
            ORDER BY
                CASE
                    WHEN (
                        ship_name_en = %s OR ship_name_local = %s
                    )
                    THEN 0 ELSE 1
                END,
                id
        """
        params: tuple[Any, ...] = (
            query,
            query,
            query,
            query,
            query,
            query,
            query,
            query,
            query,
        )
        cursor.execute(sql, params)
        fittings = [
            fitting
            for fitting in cursor.fetchall()
            if any(
                _fitting_search_match(query, candidate)
                for candidate in (
                    fitting.get("ship_name_en", ""),
                    fitting.get("ship_name_local", ""),
                    fitting.get("fitting_name", ""),
                )
            )
        ]
        if limit is not None:
            fittings = fittings[:limit]
        if not fittings:
            return fittings

        fitting_ids = [int(fitting["id"]) for fitting in fittings]
        placeholders = ", ".join(["%s"] * len(fitting_ids))
        cursor.execute(
            f"""
            SELECT fitting_id, slot_name, quantity, item_name_en, item_name_local
            FROM `{items_table}`
            WHERE fitting_id IN ({placeholders})
            ORDER BY fitting_id, item_index
            """,
            fitting_ids,
        )
        items_by_fitting: dict[int, list[dict[str, Any]]] = {}
        for item in cursor.fetchall():
            items_by_fitting.setdefault(int(item["fitting_id"]), []).append(
                {
                    "slot_name": item["slot_name"],
                    "quantity": item["quantity"],
                    "item_name_en": item["item_name_en"],
                    "item_name_local": item["item_name_local"],
                }
            )
        for fitting in fittings:
            fitting["items"] = items_by_fitting.get(int(fitting["id"]), [])
        return fittings


def filter_fittings_by_ship_query(
    fittings: list[dict[str, Any]], query: str
) -> list[dict[str, Any]]:
    """Prefer exact ship matches when a natural-language query is broad."""
    normalized = " ".join((query or "").casefold().split())
    exact: list[dict[str, Any]] = []
    for fitting in fittings:
        aliases = {
            str(fitting.get("ship_name_en", "") or "").casefold().strip(),
            str(fitting.get("ship_name_local", "") or "").casefold().strip(),
        }
        aliases.update(
            alias[:-1] for alias in list(aliases) if alias.endswith("级")
        )
        if any(
            alias and _contains_search_term(normalized, alias)
            for alias in aliases
        ):
            exact.append(fitting)
    return exact or fittings


def find_fittings_for_simulation(
    query: str,
    conversation_history: list[dict[str, Any]] | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find likely fitting candidates for metric analysis.

    Exact ship names are preferred. When the user only gives a style such as
    "PVE shield missile", rank the full fitting catalog by matching fitting,
    ship, and module text.
    """
    parts = [str(query or "")]
    for message in (conversation_history or [])[-8:]:
        if message.get("role") in {"user", "assistant"}:
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
    combined = "\n".join(parts)
    all_rows = find_all_fittings()
    if not all_rows:
        return []

    explicit = filter_fittings_by_ship_query(all_rows, combined)
    # filter_fittings_by_ship_query returns all rows when no exact ship was
    # found, so detect exact aliases independently before deciding the path.
    exact_rows: list[dict[str, Any]] = []
    for row in all_rows:
        aliases = [
            str(row.get("ship_name_en", "") or ""),
            str(row.get("ship_name_local", "") or ""),
        ]
        if any(
            alias and _contains_search_term(combined.casefold(), alias.casefold())
            for alias in aliases
        ):
            exact_rows.append(row)
    if exact_rows:
        return exact_rows[: max(1, limit)]

    terms = [
        term.casefold()
        for term in re.findall(r"[A-Za-z0-9]{3,}|[\u4e00-\u9fff]{2,}", combined)
    ]
    stop_words = {
        "检查一下", "这个用户", "如果用", "这套方案", "是多少", "然后",
        "数值是多少", "帮我", "看看", "现在", "技能", "调用", "的话",
    }
    terms = [term for term in terms if term not in stop_words]

    def score(row: dict[str, Any]) -> tuple[int, int, int, int]:
        haystack = " ".join(
            [
                str(row.get("fitting_name", "") or ""),
                str(row.get("ship_name_en", "") or ""),
                str(row.get("ship_name_local", "") or ""),
                str(row.get("description", "") or ""),
                " ".join(
                    str(item.get("item_name_en", "") or "")
                    + " "
                    + str(item.get("item_name_local", "") or "")
                    for item in row.get("items", [])
                ),
            ]
        ).casefold()
        matched = sum(1 for term in terms if term in haystack)
        style_bonus = 0
        if re.search(r"(pve|刷怪|速刷|异常|任务)", combined, re.IGNORECASE) and re.search(
            r"(pve|刷怪|速刷|异常|任务)", haystack, re.IGNORECASE
        ):
            style_bonus += 3
        if re.search(r"(盾|护盾|shield)", combined, re.IGNORECASE) and re.search(
            r"(盾|护盾|shield)", haystack, re.IGNORECASE
        ):
            style_bonus += 2
        if re.search(r"(导弹|missile)", combined, re.IGNORECASE) and re.search(
            r"(导弹|missile)", haystack, re.IGNORECASE
        ):
            style_bonus += 2
        if re.search(r"(盾导弹|盾.*导弹|导弹.*盾|shield.*missile|missile.*shield)", combined, re.IGNORECASE):
            if re.search(r"(shield|护盾|盾|shield booster|shield hardener|shield extender)", haystack, re.IGNORECASE) and re.search(
                r"(missile|导弹|launcher|发射器)", haystack, re.IGNORECASE
            ):
                style_bonus += 8
        has_shield = bool(
            re.search(r"(shield|护盾|盾)", haystack, re.IGNORECASE)
        )
        has_missile = bool(
            re.search(r"(missile|导弹|launcher|发射器)", haystack, re.IGNORECASE)
        )
        has_pve_style = bool(
            re.search(r"(pve|刷怪|速刷|异常|任务)", haystack, re.IGNORECASE)
        )
        fit_kind = 2 if has_shield and has_missile and has_pve_style else (
            1 if has_shield and has_missile else 0
        )
        return (
            fit_kind,
            style_bonus,
            matched,
            -int(row.get("id", 0) or 0),
        )

    ranked = sorted(all_rows, key=score, reverse=True)
    return [row for row in ranked if score(row)[0] > 0][: max(1, limit)]


def find_fitting_targets_in_text(
    text: str, max_targets: int = 6
) -> list[dict[str, str]]:
    """Find fitting ships mentioned in research text, including local aliases."""
    normalized_text = " ".join((text or "").casefold().split())
    if not normalized_text:
        return []
    fittings_table, _ = _identifiers()
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT DISTINCT id, ship_name_en, ship_name_local, fitting_name
            FROM `{fittings_table}`
            WHERE ship_name_en <> '' OR ship_name_local <> '' OR fitting_name <> ''
            ORDER BY id
            """
        )
        rows = cursor.fetchall()

    targets: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        local_name = str(row.get("ship_name_local", "") or "").strip()
        english_name = str(row.get("ship_name_en", "") or "").strip()
        fitting_name = str(row.get("fitting_name", "") or "").strip()
        aliases = [local_name, english_name, fitting_name]
        aliases.extend(
            re.findall(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{2,}", fitting_name)
        )
        if local_name.endswith("级"):
            aliases.append(local_name[:-1])
        matched_alias = next(
            (
                alias
                for alias in aliases
                if alias and _contains_search_term(normalized_text, alias.casefold())
            ),
            "",
        )
        if not matched_alias:
            continue
        target = local_name or english_name
        key = target.casefold()
        if not target or key in seen:
            continue
        targets.append(
            {
                "query": target,
                "reason": f"研究结果中出现“{matched_alias}”，来自装配库舰船/装配名匹配",
            }
        )
        seen.add(key)
        if len(targets) >= max_targets:
            break
    return targets


def _fitting_search_match(query: str, candidate: str | None) -> bool:
    """Match names in either direction without treating YST as part of Catalyst."""
    query = " ".join((query or "").casefold().split())
    candidate = " ".join((candidate or "").casefold().split())
    if not query or not candidate:
        return False
    if _contains_search_term(query, candidate):
        return True
    if candidate.endswith("级") and _contains_search_term(query, candidate[:-1]):
        return True
    return _contains_search_term(candidate, query)


def _contains_search_term(text: str, term: str) -> bool:
    if term not in text:
        return False
    if term.isascii() and re.fullmatch(r"[a-z0-9]+", term):
        return re.search(
            rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
            text,
            re.IGNORECASE,
        ) is not None
    return True


def format_context(fittings: list[dict[str, Any]]) -> str:
    lines = [
        "以下是从 MySQL 的 EVE 装配数据库读取的权威装配数据。",
        "用户询问舰船装配时，优先依据这些数据回答；不要臆造数据库中不存在的装备。",
    ]
    for number, fitting in enumerate(fittings, 1):
        ship = fitting["ship_name_local"] or fitting["ship_name_en"]
        if fitting["ship_name_en"] and fitting["ship_name_local"]:
            ship = f"{ship} ({fitting['ship_name_en']})"
        lines.append(
            f"\n装配 {number}：{fitting['fitting_name']} | 舰船：{ship} | 来源：{fitting['source_file']}"
        )
        if fitting.get("description"):
            lines.append(f"说明：{fitting['description']}")
        grouped: dict[str, list[str]] = {}
        for item in fitting["items"]:
            name = item["item_name_local"] or item["item_name_en"]
            if item["item_name_en"] and item["item_name_local"]:
                name = f"{name} ({item['item_name_en']})"
            quantity = item["quantity"] if item["quantity"] > 1 else ""
            grouped.setdefault(item["slot_name"] or "未分类", []).append(
                f"{name}{f' x{quantity}' if quantity else ''}"
            )
        for slot, items in grouped.items():
            lines.append(f"- {slot}: {'；'.join(items)}")
    return "\n".join(lines)


def find_all_fittings() -> list[dict[str, Any]]:
    """Load the complete fitting catalog with slot items."""
    fittings_table, items_table = _identifiers()
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT id, source_file, fitting_name, ship_name_en, ship_name_local,
                   description
            FROM `{fittings_table}`
            ORDER BY id
            """
        )
        fittings = list(cursor.fetchall())
        if not fittings:
            return []
        ids = [int(row["id"]) for row in fittings]
        placeholders = ", ".join(["%s"] * len(ids))
        cursor.execute(
            f"""
            SELECT fitting_id, slot_name, quantity, item_name_en, item_name_local
            FROM `{items_table}`
            WHERE fitting_id IN ({placeholders})
            ORDER BY fitting_id, item_index
            """,
            ids,
        )
        items_by_fitting: dict[int, list[dict[str, Any]]] = {}
        for item in cursor.fetchall():
            items_by_fitting.setdefault(int(item["fitting_id"]), []).append(
                {
                    "slot_name": item["slot_name"],
                    "quantity": item["quantity"],
                    "item_name_en": item["item_name_en"],
                    "item_name_local": item["item_name_local"],
                }
            )
        for fitting in fittings:
            fitting["items"] = items_by_fitting.get(int(fitting["id"]), [])
        return fittings
