"""MySQL-backed access to all EVE SDE JSONL datasets."""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pymysql
from pymysql.cursors import DictCursor


DEFAULT_ENV_FILE = Path(os.getenv("EVE_ENV_FILE", ".env"))
DEFAULT_SDE_DIR = Path(os.getenv("EVE_SDE_DIR", "data/sde"))
DEFAULT_DATABASE = "EVE"
TABLE_NAME = "eve_sde_records"


def _load_dotenv(path: Path = DEFAULT_ENV_FILE) -> None:
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


@contextmanager
def connection() -> Iterator[pymysql.connections.Connection]:
    conn = pymysql.connect(**_db_config())
    try:
        yield conn
    finally:
        conn.close()


def _safe_dataset(filename: str) -> str:
    return Path(filename).stem.lower().replace("-", "_")


def _record_key(record: dict[str, Any], line_number: int) -> str:
    value = record.get("_key")
    if value is None:
        for candidate in (
            "typeID",
            "groupID",
            "blueprintTypeID",
            "solarSystemID",
            "regionID",
            "constellationID",
            "stationID",
            "characterID",
            "corporationID",
            "factionID",
        ):
            if record.get(candidate) is not None:
                value = record[candidate]
                break
    return str(value if value is not None else line_number)


def _name_value(record: dict[str, Any], language: str) -> str:
    for key in ("name", "itemName", "title"):
        value = record.get(key)
        if isinstance(value, dict):
            result = value.get(language)
            if result is not None:
                return str(result)
        elif isinstance(value, str) and language == "en":
            return value
    return ""


def _search_text(record: dict[str, Any]) -> str:
    values: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(record)
    return " ".join(values)[:16000]


def ensure_schema(conn: pymysql.connections.Connection) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
                dataset VARCHAR(96) NOT NULL,
                record_key VARCHAR(128) NOT NULL,
                name_zh VARCHAR(512) NOT NULL DEFAULT '',
                name_en VARCHAR(512) NOT NULL DEFAULT '',
                search_text TEXT NOT NULL,
                raw_json LONGTEXT NOT NULL,
                source_file VARCHAR(255) NOT NULL,
                source_line INT UNSIGNED NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (dataset, record_key),
                KEY idx_sde_dataset_zh (dataset, name_zh(191)),
                KEY idx_sde_dataset_en (dataset, name_en(191)),
                KEY idx_sde_source (source_file)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )


def import_sde(source_dir: Path = DEFAULT_SDE_DIR, batch_size: int = 500) -> dict[str, Any]:
    paths = sorted(source_dir.glob("*.jsonl"))
    total = 0
    datasets: dict[str, int] = {}
    with connection() as conn:
        ensure_schema(conn)
        with conn.cursor() as cursor:
            for path in paths:
                dataset = _safe_dataset(path.name)
                batch: list[tuple[Any, ...]] = []
                count = 0
                with path.open("r", encoding="utf-8", errors="replace") as stream:
                    for line_number, raw_line in enumerate(stream, 1):
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(record, dict):
                            continue
                        batch.append(
                            (
                                dataset,
                                _record_key(record, line_number),
                                _name_value(record, "zh"),
                                _name_value(record, "en"),
                                _search_text(record),
                                json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                                path.name,
                                line_number,
                            )
                        )
                        count += 1
                        if len(batch) >= batch_size:
                            _upsert_batch(cursor, batch)
                            batch.clear()
                    if batch:
                        _upsert_batch(cursor, batch)
                datasets[dataset] = count
                total += count
                conn.commit()
                print(f"{dataset}: {count}")
    return {"files": len(paths), "records": total, "datasets": datasets}


def _upsert_batch(cursor: Any, batch: list[tuple[Any, ...]]) -> None:
    cursor.executemany(
        f"""
        INSERT INTO `{TABLE_NAME}`
            (dataset, record_key, name_zh, name_en, search_text,
             raw_json, source_file, source_line)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            name_zh=VALUES(name_zh),
            name_en=VALUES(name_en),
            search_text=VALUES(search_text),
            raw_json=VALUES(raw_json),
            source_file=VALUES(source_file),
            source_line=VALUES(source_line)
        """,
        batch,
    )


def dataset_names() -> list[str]:
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            f"SELECT DISTINCT dataset FROM `{TABLE_NAME}` ORDER BY dataset"
        )
        return [row["dataset"] for row in cursor.fetchall()]


def query_static_data(
    query: str,
    dataset: str | None = None,
    record_key: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    query = query.strip()
    if dataset in (None, "types") and not record_key:
        catalog = _query_ship_catalog(query, limit)
        if catalog is not None:
            return catalog
    if dataset in (None, "types") and _is_webifier_catalog_query(query):
        return _query_webifier_catalog(limit)
    search_query = _extract_static_subject(query) if dataset == "types" else query
    if not record_key and search_query:
        from eve_lightrag.eve_names import resolve_name

        official = resolve_name(search_query, _safe_dataset(dataset) if dataset else None)
        if official:
            dataset = official["dataset"]
            record_key = official["entity_id"]
    conditions: list[str] = []
    params: list[Any] = []
    if dataset:
        conditions.append("dataset = %s")
        params.append(_safe_dataset(dataset))
    if record_key:
        conditions.append("record_key = %s")
        params.append(record_key.strip())
    elif search_query:
        if search_query.isdigit():
            conditions.append(
                "(record_key = %s OR LOWER(name_zh) LIKE LOWER(%s)"
                " OR LOWER(name_en) LIKE LOWER(%s)"
                " OR LOWER(search_text) LIKE LOWER(%s))"
            )
            wildcard = f"%{search_query}%"
            params.extend((search_query, wildcard, wildcard, wildcard))
        else:
            conditions.append(
                "((name_zh <> '' AND LOCATE(LOWER(name_zh), LOWER(%s)) > 0)"
                " OR (name_en <> '' AND LOCATE(LOWER(name_en), LOWER(%s)) > 0)"
                " OR LOWER(name_zh) LIKE LOWER(%s) OR LOWER(name_en) LIKE LOWER(%s)"
                " OR LOWER(search_text) LIKE LOWER(%s))"
            )
            wildcard = f"%{search_query}%"
            params.extend(
                (search_query, search_query, wildcard, wildcard, wildcard)
            )
    where = " AND ".join(conditions) if conditions else "1=1"
    limit = max(1, min(int(limit), 100))
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT dataset, record_key, name_zh, name_en, raw_json, source_file,
                   source_line
            FROM `{TABLE_NAME}`
            WHERE {where}
            ORDER BY
                CASE
                    WHEN name_zh = %s OR name_en = %s THEN 0
                    WHEN name_zh LIKE %s OR name_en LIKE %s THEN 1
                    ELSE 2
                END,
                dataset, record_key
            LIMIT %s
            """,
            params
            + [
                search_query,
                search_query,
                f"{search_query}%",
                f"{search_query}%",
                limit,
            ],
        )
        rows = cursor.fetchall()
    for row in rows:
        try:
            row["data"] = json.loads(row.pop("raw_json"))
        except (TypeError, json.JSONDecodeError):
            row["data"] = row.pop("raw_json")
    _enrich_type_rows(rows)
    return _prefer_specific_type_matches(rows, search_query)


def _query_ship_catalog(query: str, limit: int) -> list[dict[str, Any]] | None:
    """Resolve ship-class lists through SDE group IDs, not description LIKEs."""
    if not re.search(r"列表|清单|有哪些|哪几|四族|各族|分别|\blist\b", query, re.I):
        return None
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            f"SELECT record_key,name_en,name_zh FROM `{TABLE_NAME}` "
            "WHERE dataset='groups' AND JSON_EXTRACT(raw_json,'$.categoryID')=6"
        )
        aliases = {}
        for group in cursor.fetchall():
            for name in (group['name_en'], group['name_zh']):
                if name:
                    aliases[name.casefold()] = group['record_key']
            if group['record_key'] == '547':
                aliases['航母'] = group['record_key']
                aliases['舰队航母'] = group['record_key']
            elif group['record_key'] == '659':
                aliases['超级航空母舰'] = group['record_key']
        alternatives = []
        for name in sorted(aliases, key=len, reverse=True):
            escaped = re.escape(name)
            alternatives.append(r'(?<![a-z])' + escaped + r's?(?![a-z])' if name.isascii() else escaped)
        matches = re.finditer('|'.join(alternatives), query.casefold())
        group_ids = {aliases[m.group().removesuffix('s')] if m.group() not in aliases else aliases[m.group()] for m in matches}
        if not group_ids:
            return None
        placeholders = ','.join(['%s'] * len(group_ids))
        race_filter = """AND JSON_EXTRACT(t.raw_json,'$.raceID') IN (1,2,4,8)
            AND (JSON_EXTRACT(t.raw_json,'$.factionID') IS NULL OR
                 JSON_EXTRACT(t.raw_json,'$.factionID') IN (500001,500002,500003,500004))""" if re.search(r'四族|各族', query) else ''
        cursor.execute(
            f"""SELECT t.dataset,t.record_key,t.name_zh,t.name_en,t.raw_json,
                       t.source_file,t.source_line,g.name_zh AS ship_class_zh,
                       r.name_zh AS race_zh
                FROM `{TABLE_NAME}` t
                LEFT JOIN `{TABLE_NAME}` g ON g.dataset='groups'
                    AND g.record_key=JSON_UNQUOTE(JSON_EXTRACT(t.raw_json,'$.groupID'))
                LEFT JOIN `{TABLE_NAME}` r ON r.dataset='races'
                    AND r.record_key=JSON_UNQUOTE(JSON_EXTRACT(t.raw_json,'$.raceID'))
                WHERE t.dataset='types'
                    AND CAST(JSON_EXTRACT(t.raw_json,'$.groupID') AS UNSIGNED) IN ({placeholders})
                    AND JSON_UNQUOTE(JSON_EXTRACT(t.raw_json,'$.published'))='true'
                    {race_filter}
                ORDER BY g.record_key,r.record_key,t.record_key LIMIT %s""",
            [*sorted(group_ids), max(1, min(int(limit), 100))],
        )
        rows = cursor.fetchall()
    for row in rows:
        row['data'] = json.loads(row.pop('raw_json'))
        row['ship_catalog'] = True
    return rows


def _is_webifier_catalog_query(query: str) -> bool:
    """Recognize broad webifier type questions before semantic overmatching."""
    normalized = " ".join((query or "").casefold().split())
    has_webifier = bool(
        re.search(
            r"(网子|停滞缠绕|停滞缠绕光束|stasis\s+webifier)",
            normalized,
            re.IGNORECASE,
        )
    )
    if not has_webifier:
        return False
    return not bool(
        re.search(
            r"(属性|参数|数值|强度|射程|电容|cpu|能量栅格|type\s*id|dogma|"
            r"具体型号|某个|这件|这个|哪一个|获取方式)",
            normalized,
            re.IGNORECASE,
        )
    )


def _extract_static_subject(query: str) -> str:
    """Reduce a natural-language static-data question to its item subject."""
    subject = " ".join((query or "").strip().split())
    if not subject:
        return subject
    markers = (
        "这个装备",
        "该装备",
        "此装备",
        "这个物品",
        "该物品",
        "这个模块",
        "该模块",
        "的属性",
        "属性呢",
        "有哪些类型",
        "有哪些型号",
        "有哪些分类",
        "的类型",
        "的型号",
        "安装数量",
        "装备数量",
        "装配数量",
        "能装几个",
        "最多装几个",
        "安装上限",
        "数量限制",
    )
    positions = [
        subject.find(marker)
        for marker in markers
        if subject.find(marker) > 0
    ]
    if positions:
        subject = subject[: min(positions)]
    subject = re.sub(
        r"(?:在舰船上|在船上|舰船上|船上|有没有|是否有|具体是多少|是多少|"
        r"如果有|请问|查询|查一下|呢|吗)[？?。！!，,；;].*$",
        "",
        subject,
    )
    subject = subject.strip(" \t\r\n，,。！？?：:的")
    return subject or query.strip()


def _query_webifier_catalog(limit: int) -> list[dict[str, Any]]:
    """Return only actual ship-fitting webifier modules, not related records."""
    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT dataset, record_key, name_zh, name_en, raw_json, source_file,
                   source_line
            FROM `{TABLE_NAME}`
            WHERE dataset = 'types'
              AND (
                  name_en LIKE %s
                  OR name_zh LIKE %s
              )
            ORDER BY CAST(record_key AS UNSIGNED)
            """,
            ("%Stasis Webifier%", "%停滞缠绕光束%"),
        )
        rows = cursor.fetchall()

    filtered: list[dict[str, Any]] = []
    for row in rows:
        try:
            data = json.loads(row.pop("raw_json"))
        except (TypeError, json.JSONDecodeError):
            continue
        # Group 65 is the ship module group. This excludes blueprints,
        # mutaplasmids, structures, NPC towers, and the webifier burst.
        if data.get("groupID") != 65:
            continue
        row["data"] = data
        row["webifier_catalog"] = True
        filtered.append(row)

    _enrich_type_rows(filtered)
    if limit > 0:
        return filtered[: min(int(limit), len(filtered))]
    return filtered


def _prefer_specific_type_matches(
    rows: list[dict[str, Any]], query: str
) -> list[dict[str, Any]]:
    """Avoid returning a shorter type name when a more specific name is queried."""
    normalized_query = query.casefold()
    type_rows = [row for row in rows if row.get("dataset") == "types"]
    matching_type_rows = [
        row
        for row in type_rows
        if any(
            name and name.casefold() in normalized_query
            for name in (row.get("name_zh"), row.get("name_en"))
        )
    ]
    if len(matching_type_rows) < 2:
        return rows

    def matched_name_length(row: dict[str, Any]) -> int:
        return max(
            (
                len(name)
                for name in (row.get("name_zh"), row.get("name_en"))
                if name and name.casefold() in normalized_query
            ),
            default=0,
        )

    longest = max(matched_name_length(row) for row in matching_type_rows)
    specific_rows = {
        id(row) for row in matching_type_rows if matched_name_length(row) == longest
    }
    return [
        row
        for row in rows
        if row.get("dataset") != "types" or id(row) in specific_rows
    ]


def _enrich_type_rows(rows: list[dict[str, Any]]) -> None:
    """Attach readable Dogma attributes to item/ship type records."""
    type_keys = [
        str(row["record_key"])
        for row in rows
        if row.get("dataset") == "types" and str(row.get("record_key", "")).isdigit()
    ]
    if not type_keys:
        return

    with connection() as conn, conn.cursor() as cursor:
        meta_group_ids: set[str] = set()
        for row in rows:
            if row.get("dataset") != "types" or not isinstance(row.get("data"), dict):
                continue
            meta_group_id = row["data"].get("metaGroupID")
            if meta_group_id is not None:
                meta_group_ids.add(str(meta_group_id))

        meta_groups_by_id: dict[str, dict[str, Any]] = {}
        if meta_group_ids:
            meta_placeholders = ", ".join(["%s"] * len(meta_group_ids))
            cursor.execute(
                f"""
                SELECT record_key, raw_json
                FROM `{TABLE_NAME}`
                WHERE dataset = 'metagroups' AND record_key IN ({meta_placeholders})
                """,
                list(meta_group_ids),
            )
            for raw_row in cursor.fetchall():
                try:
                    meta_groups_by_id[str(raw_row["record_key"])] = json.loads(
                        raw_row["raw_json"]
                    )
                except (TypeError, json.JSONDecodeError):
                    continue

        placeholders = ", ".join(["%s"] * len(type_keys))
        cursor.execute(
            f"""
            SELECT record_key, raw_json
            FROM `{TABLE_NAME}`
            WHERE dataset = 'typedogma' AND record_key IN ({placeholders})
            """,
            type_keys,
        )
        dogma_by_type: dict[str, dict[str, Any]] = {}
        attribute_ids: set[str] = set()
        for raw_row in cursor.fetchall():
            try:
                data = json.loads(raw_row["raw_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            dogma_by_type[str(raw_row["record_key"])] = data
            for attribute in data.get("dogmaAttributes", []):
                if isinstance(attribute, dict) and attribute.get("attributeID") is not None:
                    attribute_ids.add(str(attribute["attributeID"]))

        if not attribute_ids:
            return
        attr_placeholders = ", ".join(["%s"] * len(attribute_ids))
        cursor.execute(
            f"""
            SELECT record_key, raw_json
            FROM `{TABLE_NAME}`
            WHERE dataset = 'dogmaattributes' AND record_key IN ({attr_placeholders})
            """,
            list(attribute_ids),
        )
        attributes_by_id: dict[str, dict[str, Any]] = {}
        for raw_row in cursor.fetchall():
            try:
                attributes_by_id[str(raw_row["record_key"])] = json.loads(
                    raw_row["raw_json"]
                )
            except (TypeError, json.JSONDecodeError):
                continue

    for row in rows:
        if row.get("dataset") != "types":
            continue
        if isinstance(row.get("data"), dict):
            meta_group_id = row["data"].get("metaGroupID")
            meta_group = meta_groups_by_id.get(str(meta_group_id))
            if meta_group:
                row["meta_group"] = {
                    "metaGroupID": meta_group_id,
                    "name_zh": meta_group.get("name", {}).get("zh", ""),
                    "name_en": meta_group.get("name", {}).get("en", ""),
                }
        dogma = dogma_by_type.get(str(row["record_key"]))
        if not dogma:
            continue
        readable_attributes: list[dict[str, Any]] = []
        for attribute in dogma.get("dogmaAttributes", []):
            if not isinstance(attribute, dict):
                continue
            attribute_id = str(attribute.get("attributeID", ""))
            definition = attributes_by_id.get(attribute_id, {})
            display_name = definition.get("displayName", {})
            readable_attributes.append(
                {
                    "attributeID": attribute.get("attributeID"),
                    "name": definition.get("name", ""),
                    "displayName_zh": display_name.get("zh", ""),
                    "displayName_en": display_name.get("en", ""),
                    "value": attribute.get("value"),
                    "unitID": definition.get("unitID"),
                    "description_zh": (
                        definition.get("tooltipDescription", {}).get("zh")
                        or definition.get("description", "")
                    ),
                }
            )
        row["dogma"] = {
            "attributes": readable_attributes,
            "effects": dogma.get("dogmaEffects", []),
        }


def format_static_context(rows: list[dict[str, Any]]) -> str:
    hard_facts: list[str] = []
    for row in rows:
        if row.get("dataset") != "types":
            continue
        data = row.get("data")
        if not isinstance(data, dict):
            continue
        dogma = row.get("dogma")
        if not isinstance(dogma, dict):
            dogma = data.get("dogma")
        attributes = dogma.get("attributes", []) if isinstance(dogma, dict) else []
        max_group_fitted = next(
            (
                attr
                for attr in attributes
                if isinstance(attr, dict)
                and (
                    str(attr.get("attributeID")) == "1544"
                    or attr.get("name") == "maxGroupFitted"
                )
            ),
            None,
        )
        if max_group_fitted is None:
            continue
        value = max_group_fitted.get("value")
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        display_name = row.get("name_zh") or row.get("name_en") or row.get("record_key")
        hard_facts.append(
            f"装备：{display_name}；字段：maxGroupFitted（attributeID=1544，"
            f"该武器组所允许的最大装备数量）；value：{value}；"
            f"结论：一艘舰船最多安装{value}个该装备组模块。"
        )

    lines = [
        "以下是从 MySQL EVE 静态数据 API 查询得到的权威数据。",
        "只能依据这些结果回答静态数据问题；如果结果中没有字段，不要自行编造。",
        "安装数量问题必须读取 dogma.attributes 中的 maxGroupFitted "
        "（attributeID=1544，中文名为“该武器组所允许的最大装备数量”）。"
        "该字段的 value 是硬性上限，必须原样采用；不得把 warpScrambleStrength "
        "（跃迁干扰强度）或其他属性 value 当成安装数量，也不得凭记忆推测。",
    ]
    if hard_facts:
        lines.extend(
            [
                "",
                "【硬性数据库事实：优先使用】",
                "以下事实来自 MySQL，回答安装数量问题时必须直接采用，不得被后面的原始 JSON、知识库或模型记忆覆盖：",
                *hard_facts,
            ]
        )
    catalog = [row for row in rows if row.get('ship_catalog')]
    if catalog:
        lines.extend(['', '【SDE 官方舰船清单：必须使用这些舰名，不得自行音译】'])
        for row in catalog:
            lines.append(f"{row['race_zh']} | {row['ship_class_zh']} | {row['name_zh']} | {row['name_en']} | typeID={row['record_key']}")
    for index, row in enumerate(rows, 1):
        lines.append(
            f"\n数据 {index} | dataset={row['dataset']} | key={row['record_key']} "
            f"| 中文名={row['name_zh']} | 英文名={row['name_en']} "
            f"| source={row['source_file']}"
        )
        data = dict(row["data"]) if isinstance(row["data"], dict) else row["data"]
        if row.get('ship_catalog'):
            data = {
                'typeID': row['record_key'], 'name_zh': row['name_zh'],
                'name_en': row['name_en'], 'groupID': data.get('groupID'),
                'raceID': data.get('raceID'), 'race_zh': row['race_zh'],
                'ship_class_zh': row['ship_class_zh'],
                'description_zh': (data.get('description') or {}).get('zh', ''),
            }
        if row.get("webifier_catalog") and isinstance(data, dict):
            # A catalog query needs every module name and classification. Full
            # Dogma payloads make the prompt huge and cause the answer model to
            # stop after the first few modules.
            data = {
                "typeID": data.get("typeID", row["record_key"]),
                "typeName": data.get("name", {}),
                "groupID": data.get("groupID"),
                "metaGroupID": data.get("metaGroupID"),
                "meta_group": row.get("meta_group"),
            }
        if isinstance(data, dict) and row.get("dogma") and not row.get(
            "webifier_catalog"
        ):
            data["dogma"] = row["dogma"]
        if isinstance(data, dict) and row.get("meta_group"):
            data["meta_group"] = row["meta_group"]
        lines.append(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        if isinstance(data, dict):
            attributes = (data.get("dogma") or {}).get("attributes", [])
            max_group_fitted = next(
                (
                    attr
                    for attr in attributes
                    if attr.get("attributeID") == 1544
                    or attr.get("name") == "maxGroupFitted"
                ),
                None,
            )
            if max_group_fitted is not None:
                value = max_group_fitted.get("value")
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                lines.append(
                    "【安装数量硬性结论】"
                    f"{row['name_zh'] or row['name_en']} 的 "
                    "maxGroupFitted（最大装备数量）="
                    f"{value}。"
                    f"一艘舰船最多只能安装{value}个此装备组模块；"
                    "该结论优先于其他属性和模型记忆。"
                )
    webifier_rows = [
        row
        for row in rows
        if isinstance(row.get("data"), dict)
        and row["data"].get("groupID") == 65
        and "Stasis Webifier" in str(row.get("name_en", ""))
    ]
    if webifier_rows:
        meta_ids = {
            row["data"].get("metaGroupID")
            for row in webifier_rows
            if row["data"].get("metaGroupID") is not None
        }
        lines.extend(
            [
                "",
                "【网子类型判定规则】",
                "以上清单只包含 groupID=65 的停滞缠绕模块。",
                f"本次结果中的 metaGroupID：{sorted(meta_ids, key=str)}。",
                "本次结果没有返回 metaGroupID=6；未返回的分类不得列入类型清单。",
                "metaGroupID=5 的分类名称是 Officer/官员，"
                "metaGroupID=15 的分类名称是 Abyssal/深渊，必须按数据库名称原样表达。",
            ]
        )
    return "\n".join(lines)


_STATIC_INTENT_PATTERNS = (
    r"静态数据",
    r"SDE",
    r"type\s*id",
    r"物品信息",
    r"装备属性",
    r"舰船属性",
    r"舰船资料",
    r"舰船信息",
    r"星系信息",
    r"星系属性",
    r"蓝图信息",
    r"制造材料",
    r"技能信息",
    r"技能属性",
    r"属性(?:呢|信息|数据|数值|参数|有哪些|如何|是多少)?",
    r"(?:安装|装配|装备).{0,8}(?:数量|上限|限制|几个|多少个)",
    r"(?:数量|上限|限制|几个|多少个).{0,8}(?:安装|装配|装备)",
    r"\battributes?\b",
    r"\b(?:item|module|ship)\s+stats?\b",
    r"\bproperties\b",
    r"dogma",
    r"属性是什么",
    r"(?:网子|停滞缠绕|停滞缠绕光束).{0,12}(?:类型|型号|分类|变体|有哪些)",
    r"是什么物品",
    r"是什么舰船",
    r"属于什么分类",
    r"\btype\s*id\b",
    r"\bblueprint\b",
    r"\bship\s+stats?\b",
    r"\bitem\s+info\b",
    r"\bsolar\s+system\b",
)


def is_static_data_query(query: str) -> bool:
    if not query or not query.strip():
        return False
    normalized = " ".join(query.casefold().split())
    return any(
        re.search(pattern, normalized, re.IGNORECASE)
        for pattern in _STATIC_INTENT_PATTERNS
    )


def preferred_static_dataset(query: str) -> str | None:
    """Choose the SDE dataset for item/ship/module questions.

    Generic full-text matching can otherwise return category or market-group
    records whose search text happens to contain the requested words. Concrete
    EVE item and fitting-property questions belong to the types dataset.
    """
    normalized = " ".join((query or "").casefold().split())
    if re.search(
        r"(装备|物品|模块|舰船|飞船|网子|停滞缠绕|属性|安装|装配数量|"
        r"能装|最多装|module|item|ship|dogma|attribute|type\s*id)",
        normalized,
        re.IGNORECASE,
    ):
        return "types"
    return None
