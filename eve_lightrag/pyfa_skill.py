"""Pyfa/Eos-backed EVE fitting simulation skill.

This adapter imports the LGPL-licensed ``eos`` calculation library from the
local Pyfa checkout. It deliberately does not import Pyfa's wx GUI or copy
Pyfa source into the LightRAG project.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import threading
import types
from copy import deepcopy
from math import sqrt
from pathlib import Path
from typing import Any


PYFA_ROOT = Path(os.getenv("PYFA_ROOT", "vendor/Pyfa"))
EXTENSION_ROOT = Path(__file__).resolve().parent.parent
PYFA_PYTHON = PYFA_ROOT / ".venv" / "bin" / "python"
_EOS_LOCK = threading.RLock()
_EOS_READY = False

if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))


def _ensure_eos() -> tuple[Any, Any, Any, Any]:
    """Load Eos once without requiring wxPython."""
    global _EOS_READY
    if str(PYFA_ROOT) not in sys.path:
        sys.path.insert(0, str(PYFA_ROOT))
    if str(EXTENSION_ROOT) not in sys.path:
        sys.path.insert(0, str(EXTENSION_ROOT))

    # eos.db imports Pyfa's root config through the migration module. Only
    # Colour is needed during config import; the rest of the GUI is irrelevant.
    if "wx" not in sys.modules:
        wx_stub = types.ModuleType("wx")

        class Colour:
            def __init__(self, *args: Any) -> None:
                self.args = args

        wx_stub.Colour = Colour
        sys.modules["wx"] = wx_stub

    eos = importlib.import_module("eos")
    db = importlib.import_module("eos.db")
    fit_module = importlib.import_module("eos.saveddata.fit")
    ship_module = importlib.import_module("eos.saveddata.ship")
    module_module = importlib.import_module("eos.saveddata.module")

    if not _EOS_READY:
        saveddata_dir = PYFA_ROOT / "saveddata"
        saveddata_dir.mkdir(parents=True, exist_ok=True)
        db.saveddata_meta.create_all(db.saveddata_engine)
        _EOS_READY = True
    return db, fit_module, ship_module, module_module


def _json_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return str(value)


def _resolve_type_id(name_or_id: Any) -> tuple[int | None, str]:
    if isinstance(name_or_id, int) or (
        isinstance(name_or_id, str) and name_or_id.strip().isdigit()
    ):
        type_id = int(name_or_id)
        return type_id, str(type_id)

    name = str(name_or_id or "").strip()
    if not name:
        return None, ""

    from eve_lightrag.eve_names import resolve_name

    official = resolve_name(name, "types")
    if official is not None:
        return int(official["entity_id"]), official["name_en"] or official["name_zh"]

    # Pyfa fitting exports already contain English type names. Resolve those
    # directly from the local Eos database before doing a slower MySQL alias
    # lookup.
    try:
        eos_db = importlib.import_module("eos.db")
        local_item = eos_db.getItem(name)
        if local_item is None:
            item_model = importlib.import_module("eos.db.gamedata.item").Item
            local_item = (
                eos_db.get_gamedata_session()
                .query(item_model)
                .filter(item_model.typeNameEn == name)
                .first()
            )
        if local_item is not None:
            return int(local_item.ID), str(local_item.name or local_item.typeName)
    except Exception:
        pass

    # Reuse the existing MySQL SDE name matching, including Chinese aliases.
    from eve_lightrag.eve_static_data import query_static_data

    rows = query_static_data(name, "types", None, 20)
    ranked = sorted(
        rows,
        key=lambda row: (
            0
            if name.casefold()
            in {
                str(row.get("name_zh", "")).casefold(),
                str(row.get("name_en", "")).casefold(),
            }
            else 1,
            0 if (row.get("data") or {}).get("published", False) else 1,
            1 if "blueprint" in str(row.get("name_en", "")).casefold() else 0,
        ),
    )
    if not ranked:
        return None, name
    row = ranked[0]
    return int(row["record_key"]), str(row.get("name_zh") or row.get("name_en") or name)


def _normalize_module_entries(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, int)):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("槽位装备必须是字符串、typeID 或数组")

    entries: list[dict[str, Any]] = []
    for entry in value:
        if isinstance(entry, (str, int)):
            entries.append({"name": entry, "state": "online"})
            continue
        if not isinstance(entry, dict):
            raise ValueError("装备项必须是字符串、typeID 或对象")
        name = entry.get("name", entry.get("type_id", entry.get("typeID")))
        if name is None:
            raise ValueError("装备对象缺少 name 或 typeID")
        quantity = int(entry.get("quantity", 1))
        if quantity < 1 or quantity > 32:
            raise ValueError("单个装备 quantity 必须在 1 到 32 之间")
        for _ in range(quantity):
            normalized = {
                "name": name,
                "state": str(entry.get("state", "online")).casefold(),
            }
            if entry.get("charge") is not None:
                normalized["charge"] = entry["charge"]
            entries.append(normalized)
    return entries


def _metric(fit: Any, attr: str) -> Any:
    value = fit.ship.getModifiedItemAttr(attr)
    return _json_value(value)


def _damage_map(damage: Any) -> dict[str, Any]:
    return {
        "em": _json_value(damage.em),
        "thermal": _json_value(damage.thermal),
        "kinetic": _json_value(damage.kinetic),
        "explosive": _json_value(damage.explosive),
        "pure": _json_value(damage.pure),
        "total": _json_value(damage.total),
    }


def _charge_damage_profile(charge: Any) -> dict[str, Any]:
    profile = {
        "em": charge.getAttribute("emDamage") or 0,
        "thermal": charge.getAttribute("thermalDamage") or 0,
        "kinetic": charge.getAttribute("kineticDamage") or 0,
        "explosive": charge.getAttribute("explosiveDamage") or 0,
    }
    profile["total"] = sum(float(value or 0) for value in profile.values())
    return _json_value(profile)


def _damage_charge_candidates(module: Any) -> list[Any]:
    try:
        charges = list(module.getValidCharges())
    except Exception:
        return []
    damaging = [
        charge
        for charge in charges
        if float(_charge_damage_profile(charge).get("total", 0) or 0) > 0
    ]
    return sorted(
        damaging,
        key=lambda charge: (
            float(_charge_damage_profile(charge).get("total", 0) or 0),
            str(charge.name),
        ),
        reverse=True,
    )


def _charge_summary(charge: Any) -> dict[str, Any]:
    return {
        "name": charge.name,
        "type_id": int(charge.ID),
        "group": charge.group.name,
        "damage": _charge_damage_profile(charge),
    }


def _resolve_skill_id(name_or_id: Any) -> tuple[int | None, str]:
    """Resolve a skill by Eos English name, typeID, or MySQL localized name."""
    if isinstance(name_or_id, int) or (
        isinstance(name_or_id, str) and name_or_id.strip().isdigit()
    ):
        return int(name_or_id), str(name_or_id)
    text = str(name_or_id or "").strip()
    if not text:
        return None, text
    match = re.match(r"^skill[_\s-]*id\s*:\s*(\d+)$", text, re.IGNORECASE)
    if match:
        return int(match.group(1)), text
    try:
        from eos.saveddata.character import Character

        item = Character.getSkillNameMap().get(text)
        if item is not None:
            return int(item.ID), str(item.typeName)
    except Exception:
        pass
    from eve_lightrag.eve_static_data import query_static_data

    rows = query_static_data(text, "types", None, 20)
    skill_rows = [
        row
        for row in rows
        if (row.get("data") or {}).get("categoryID") == 16
        or "skill" in str(row.get("name_en", "")).casefold()
    ]
    if not skill_rows:
        return None, text
    row = skill_rows[0]
    return int(row["record_key"]), str(row.get("name_en") or row.get("name_zh") or text)


def _build_character(
    payload: dict[str, Any],
    character_module: Any,
) -> tuple[Any | None, dict[str, Any]]:
    """Build an Eos character from all-level, ESI, and custom skill sources."""
    source = str(
        payload.get("skills_source", payload.get("skill_mode", "default"))
    ).casefold()
    if source in {"", "default", "none"}:
        return None, {"source": "default"}

    if source in {"all5", "all_5", "5", "full"}:
        character = character_module.Character("All 5", 5)
        metadata: dict[str, Any] = {"source": "all5"}
    elif source in {"all0", "all_0", "0", "zero"}:
        character = character_module.Character("All 0", 0)
        metadata = {"source": "all0"}
    elif source in {"custom", "levels"}:
        levels = payload.get("_skill_levels", payload.get("skill_levels", {}))
        if not isinstance(levels, dict):
            raise ValueError("skills_source=custom 时必须提供 skill_levels 对象")
        character = character_module.Character(
            str(payload.get("character") or "Character skills"),
            initSkills=False,
        )
        character.apiUpdateCharSheet(
            [
                {"typeID": int(skill_id), "level": int(level)}
                for skill_id, level in levels.items()
            ]
        )
        metadata = {
            "source": "custom",
            "character_name": str(payload.get("character") or "Character skills"),
            "skillpoints_by_id": {
                str(skill_id): int(value)
                for skill_id, value in (
                    payload.get("_skillpoints_by_id", {}) or {}
                ).items()
            },
        }
    elif source in {"esi", "active", "character", "real"}:
        character_ref = payload.get("character_id", payload.get("character"))
        if character_ref is None:
            raise ValueError("skills_source=esi 时必须提供 character_id 或 character")
        from eve_lightrag.eve_character_skills import get_character_skill_levels

        levels, metadata = get_character_skill_levels(
            character_ref,
            bool(payload.get("force_skill_refresh", False)),
        )
        character = character_module.Character(
            str(metadata["character_name"]), initSkills=False
        )
        character.apiUpdateCharSheet(
            [{"typeID": skill_id, "level": level} for skill_id, level in levels.items()]
        )
        metadata = {"source": "esi", **metadata}
    else:
        raise ValueError(f"不支持的 skills_source：{source}")

    custom = payload.get("skills", {})
    if custom:
        if not isinstance(custom, dict):
            raise ValueError("skills 必须是 {技能名或 skill_id:技能等级} 对象")
        custom_result: list[dict[str, Any]] = []
        for skill_name, raw_level in custom.items():
            skill_id, label = _resolve_skill_id(skill_name)
            if skill_id is None:
                raise ValueError(f"没有找到技能：{skill_name}")
            level = int(raw_level)
            if level < 0 or level > 5:
                raise ValueError(f"技能等级必须在 0 到 5 之间：{skill_name}")
            skill = character.getSkill(skill_id)
            skill.setLevel(level, ignoreRestrict=True)
            custom_result.append(
                {"skill_id": skill_id, "name": label, "level": level}
            )
        metadata["custom_overrides"] = custom_result
    return character, metadata


def _skill_points_to_level(skill_item: Any, level: int) -> int:
    """Calculate the EVE skill-point threshold using the SDE training constant."""
    if level <= 0:
        return 0
    rank = float(skill_item.getAttribute("skillTimeConstant", 1) or 1)
    # ESI's skillpoints_in_skill uses the level threshold, not the sum of
    # every training level. This yields rank-1 thresholds of 250, 1414,
    # 8000, 45255 and 256000 SP.
    return int(round(250 * rank * (sqrt(32) ** (level - 1))))


def _collect_skill_requirements(
    item: Any, requirements: dict[int, dict[str, Any]]
) -> None:
    """Collect direct and prerequisite skills recursively from an Eos item."""
    for skill_item, required_level in item.requiredSkills.items():
        skill_id = int(skill_item.ID)
        required_level = int(required_level)
        current = requirements.get(skill_id)
        if current is None or required_level > current["required_level"]:
            requirements[skill_id] = {
                "skill_id": skill_id,
                "name": str(skill_item.typeName),
                "required_level": required_level,
            }
        _collect_skill_requirements(skill_item, requirements)


def _skill_report(
    character: Any | None,
    skill_metadata: dict[str, Any],
    items: list[Any],
) -> dict[str, Any]:
    """Return required levels and missing levels/SP for a simulated fitting."""
    requirements: dict[int, dict[str, Any]] = {}
    for item in items:
        _collect_skill_requirements(item, requirements)
    rows: list[dict[str, Any]] = []
    db = importlib.import_module("eos.db")
    for skill_id, requirement in requirements.items():
        skill_item = db.getItem(skill_id)
        if skill_item is None:
            continue
        current_level = (
            int(character.getSkill(skill_id).level) if character is not None else 0
        )
        current_sp = int(
            (skill_metadata.get("skillpoints_by_id") or {}).get(str(skill_id), 0)
        )
        required_level = requirement["required_level"]
        required_sp = _skill_points_to_level(skill_item, required_level)
        missing_level = max(0, required_level - current_level)
        missing_sp = max(0, required_sp - current_sp)
        if missing_level or missing_sp:
            rows.append(
                {
                    **requirement,
                    "current_level": current_level,
                    "missing_levels": missing_level,
                    "current_sp": current_sp,
                    "required_sp": required_sp,
                    "missing_sp": missing_sp,
                }
            )
    rows.sort(key=lambda row: (-row["missing_sp"], row["name"]))
    return {
        "required_count": len(requirements),
        "missing_count": len(rows),
        "missing_skills": rows,
        "missing_sp_total": sum(row["missing_sp"] for row in rows),
    }


def fitting_to_simulation_payload(
    fitting: dict[str, Any],
    skills_source: str = "default",
    character_id: int | None = None,
) -> dict[str, Any]:
    """Convert one MySQL fitting row into the simulator request shape."""
    payload: dict[str, Any] = {
        "ship": fitting.get("ship_name_en") or fitting.get("ship_name_local"),
        "skills_source": skills_source,
    }
    if character_id is not None:
        payload["character_id"] = character_id
    slot_map = {
        "hi": "high",
        "high": "high",
        "med": "mid",
        "mid": "mid",
        "low": "low",
        "rig": "rig",
        "subsystem": "subsystem",
        "service": "service",
    }
    for slot in ("high", "mid", "low", "rig", "subsystem", "service"):
        payload[slot] = []
    for item in fitting.get("items", []):
        raw_slot = str(item.get("slot_name", "")).casefold().strip()
        slot_prefix = raw_slot.split(" ", 1)[0]
        slot = slot_map.get(slot_prefix)
        if slot is None:
            continue
        name = item.get("item_name_en") or item.get("item_name_local")
        quantity = int(item.get("quantity", 1) or 1)
        state = "online"
        lower_name = str(name or "").casefold()
        if slot == "high" or re.search(
            r"(shield booster|shield repair|shield hardener|armor repair|"
            r"armor hardener|afterburner|microwarpdrive|stasis webifier|"
            r"warp disruptor|warp scrambler|target painter|"
            r"energy neutralizer|energy nosferatu)",
            lower_name,
        ):
            state = "active"
        payload[slot].append(
            {"name": name, "quantity": quantity, "state": state}
        )
    cargo = [
        item
        for item in fitting.get("items", [])
        if str(item.get("slot_name", "")).casefold().startswith("cargo")
    ]
    charges = [
        item.get("item_name_en") or item.get("item_name_local")
        for item in cargo
        if re.search(
            r"(missile|rocket|torpedo|导弹|火箭|鱼雷)",
            str(item.get("item_name_en", ""))
            + " "
            + str(item.get("item_name_local", "")),
            re.IGNORECASE,
        )
    ]
    if charges:
        # Use the first compatible charge for every launcher. Eos validates the
        # charge/module pairing; a later fitting can explicitly override this.
        for entry in payload["high"]:
            if re.search(
                r"(launcher|炮|cannon|turret|发射器)",
                str(entry.get("name", "")),
                re.IGNORECASE,
            ):
                entry["charge"] = charges[0]
    return payload


def contains_eft_fit(text: str) -> bool:
    """Return whether text contains an EFT-style fitting block."""
    return bool(re.search(r"(?m)^\s*\[[^,\]\n]+,\s*[^\]\n]+\]\s*$", text or ""))


def _module_state_for_name(slot: str, name: str) -> str:
    lower_name = name.casefold()
    if slot == "high" or re.search(
        r"(shield booster|shield repair|armor repair|afterburner|"
        r"microwarpdrive|energy neutralizer|energy nosferatu|shield hardener)",
        lower_name,
    ):
        return "active"
    return "online"


def _parse_eft_module_line(line: str, slot: str) -> dict[str, Any]:
    parts = [part.strip() for part in line.split(",", 1)]
    entry: dict[str, Any] = {
        "name": parts[0],
        "state": _module_state_for_name(slot, parts[0]),
    }
    if len(parts) > 1 and parts[1]:
        entry["charge"] = parts[1]
    return entry


def parse_eft_fit(text: str, skills_source: str = "default") -> dict[str, Any] | None:
    """Parse a pasted EFT block into the simulator payload shape."""
    if not text:
        return None
    lines = text.splitlines()
    header_index = None
    header_match = None
    for index, line in enumerate(lines):
        match = re.match(r"^\s*\[([^,\]\n]+),\s*([^\]\n]+)\]\s*$", line)
        if match:
            header_index = index
            header_match = match
            break
    if header_index is None or header_match is None:
        return None

    payload: dict[str, Any] = {
        "ship": header_match.group(1).strip(),
        "fit_name": header_match.group(2).strip(),
        "skills_source": skills_source,
        "high": [],
        "mid": [],
        "low": [],
        "rig": [],
        "subsystem": [],
        "service": [],
    }
    section_slots = ["high", "low", "mid", "rig", "subsystem"]
    section_index = 0
    seen_module = False
    for raw_line in lines[header_index + 1 :]:
        line = raw_line.strip()
        if not line:
            if seen_module:
                section_index += 1
            continue
        # Stop once the pasted EFT block turns back into prose.
        if re.search(r"[\u4e00-\u9fff]", line) and not re.match(r"^\[[^,\]]+,", line):
            break
        if line.startswith("["):
            break
        if not re.search(r"[A-Za-z0-9]", line):
            continue
        slot = section_slots[min(section_index, len(section_slots) - 1)]
        if re.search(r"interdiction\s+nullifier", line, re.IGNORECASE):
            slot = "service"
        elif slot == "subsystem" and not re.search(
            r"\b(core|defensive|offensive|propulsion)\b", line, re.IGNORECASE
        ):
            # Extra tail lines in EFT exports are often cargo. They are not
            # fitted modules, so keep the simulator input strict.
            continue
        payload[slot].append(_parse_eft_module_line(line, slot))
        seen_module = True

    return payload if any(payload[slot] for slot in ("high", "mid", "low", "rig", "subsystem", "service")) else None


def _replace_missile_with_faction(charge: str) -> str:
    lower = charge.casefold()
    damage_map = (
        ("scourge", "Scourge"),
        ("鞭挞", "Scourge"),
        ("inferno", "Inferno"),
        ("炼狱", "Inferno"),
        ("mjolnir", "Mjolnir"),
        ("雷神", "Mjolnir"),
        ("nova", "Nova"),
        ("星爆", "Nova"),
    )
    family = "Scourge"
    for marker, replacement in damage_map:
        if marker in lower:
            family = replacement
            break
    if "heavy missile" in lower or "重型导弹" in lower:
        return f"Caldari Navy {family} Heavy Missile"
    if "heavy assault missile" in lower or "重攻击导弹" in lower:
        return f"Caldari Navy {family} Heavy Assault Missile"
    if "torpedo" in lower or "鱼雷" in lower:
        return f"Caldari Navy {family} Torpedo"
    if "cruise missile" in lower or "巡航导弹" in lower:
        return f"Caldari Navy {family} Cruise Missile"
    if "rocket" in lower or "火箭" in lower:
        return f"Caldari Navy {family} Rocket"
    return f"Caldari Navy {family} Heavy Missile"


def _apply_faction_ammo(payload: dict[str, Any]) -> bool:
    changed = False
    for entry in payload.get("high", []):
        charge = entry.get("charge")
        if not charge:
            continue
        charge_text = str(charge)
        if re.search(
            r"(fury|precision|rage|javelin|navy|missile|rocket|torpedo|导弹|火箭|鱼雷)",
            charge_text,
            re.IGNORECASE,
        ):
            new_charge = _replace_missile_with_faction(charge_text)
            if new_charge != charge_text:
                entry["charge"] = new_charge
                changed = True
    return changed


def _remove_interdiction_nullifier(payload: dict[str, Any]) -> bool:
    changed = False
    for slot in ("high", "mid", "low", "rig", "subsystem", "service"):
        before = len(payload.get(slot, []))
        payload[slot] = [
            entry
            for entry in payload.get(slot, [])
            if not re.search(
                r"interdiction\s+nullifier|拦截失效",
                str(entry.get("name", "")),
                re.IGNORECASE,
            )
        ]
        changed = changed or len(payload[slot]) != before
    return changed


def _add_cpu_coprocessor(payload: dict[str, Any]) -> str:
    low_slots = payload.setdefault("low", [])
    new_entry = {"name": "Co-Processor II", "state": "online"}
    for index, entry in enumerate(low_slots):
        if re.search(
            r"ballistic\s+control\s+system|弹道控制",
            str(entry.get("name", "")),
            re.IGNORECASE,
        ):
            replaced = str(entry.get("name", ""))
            low_slots[index] = new_entry
            return f"低槽用 Co-Processor II 替换 {replaced}"
    low_slots.append(new_entry)
    return "低槽追加 Co-Processor II"


def build_fit_variants_from_request(
    query: str,
    base_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build concrete modified payloads requested by the user."""
    normalized = query.casefold()
    wants_faction_ammo = bool(
        re.search(r"(势力导弹|海军导弹|faction\s+missile|navy\s+.*missile)", normalized)
    )
    wants_no_nullifier = bool(
        re.search(r"(不带|去掉|移除|删除|不要).{0,12}(拦截失效|中水|nullifier)", normalized)
    )
    wants_cpu = bool(
        re.search(r"(cpu\s*岗|cpu\s*挂|cpu\s*模块|cpu\s*增效|co-?processor|协处理器)", normalized)
    )

    variants: list[dict[str, Any]] = [
        {
            "variant_name": "base_eft",
            "changes": ["用户贴出的原始 EFT 装配"],
            "payload": deepcopy(base_payload),
        }
    ]
    if wants_faction_ammo:
        faction_payload = deepcopy(base_payload)
        changed = _apply_faction_ammo(faction_payload)
        variants.append(
            {
                "variant_name": "faction_ammo",
                "changes": [
                    "把发射器弹药替换为加达里海军势力导弹"
                    if changed
                    else "请求了势力导弹，但原装配没有可替换的导弹弹药"
                ],
                "payload": faction_payload,
            }
        )

    if wants_no_nullifier or wants_cpu:
        modified_payload = deepcopy(base_payload)
        changes: list[str] = []
        if wants_faction_ammo:
            if _apply_faction_ammo(modified_payload):
                changes.append("把发射器弹药替换为加达里海军势力导弹")
        if wants_no_nullifier:
            removed = _remove_interdiction_nullifier(modified_payload)
            changes.append(
                "移除 Interdiction Nullifier II"
                if removed
                else "请求移除拦截失效装置，但原装配中没有解析到该模块"
            )
        if wants_cpu:
            changes.append(_add_cpu_coprocessor(modified_payload))
        variants.append(
            {
                "variant_name": "requested_modified_fit",
                "changes": changes or ["按用户请求生成改装版本"],
                "payload": modified_payload,
            }
        )
    return variants


def simulate_fit_variants(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run many prepared fitting variants through the same Pyfa/Eos simulator."""
    results: list[dict[str, Any]] = []
    for variant in variants:
        payload = deepcopy(variant.get("payload") or {})
        try:
            simulation = simulate_fitting(payload)
            results.append(
                {
                    "skill": "pyfa_fitting_simulator",
                    "mode": "fit_variant_simulation",
                    "variant_name": variant.get("variant_name"),
                    "changes": variant.get("changes", []),
                    "payload": payload,
                    "simulation": simulation,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "skill": "pyfa_fitting_simulator",
                    "mode": "fit_variant_simulation",
                    "variant_name": variant.get("variant_name"),
                    "changes": variant.get("changes", []),
                    "payload": payload,
                    "simulation": {
                        "valid": False,
                        "errors": [{"type": "simulation_error", "error": str(exc)}],
                    },
                }
            )
    return _json_value(results)


def compare_fitting_skill_requirements(
    fittings: list[dict[str, Any]],
    character_ref: Any,
) -> list[dict[str, Any]]:
    """Compare many fitting rows against one ESI character in one Eos session."""
    with _EOS_LOCK:
        db, _fit_module, _ship_module, _module_module = _ensure_eos()
        from eos.saveddata.character import Character
        from eve_lightrag.eve_character_skills import get_character_skill_levels

        levels, metadata = get_character_skill_levels(character_ref)
        character = Character(str(metadata["character_name"]), initSkills=False)
        character.apiUpdateCharSheet(
            [{"typeID": skill_id, "level": level} for skill_id, level in levels.items()]
        )
        results: list[dict[str, Any]] = []
        for fitting in fittings:
            items: list[Any] = []
            errors: list[dict[str, Any]] = []
            ship_input = fitting.get("ship_name_en") or fitting.get("ship_name_local")
            ship_id, _ship_label = _resolve_type_id(ship_input)
            ship_item = db.getItem(ship_id) if ship_id is not None else None
            if ship_item is None:
                errors.append({"type": "ship_not_found", "name": ship_input})
            else:
                items.append(ship_item)
            for row in fitting.get("items", []):
                raw_slot = str(row.get("slot_name", "")).casefold().strip()
                if raw_slot.startswith(("cargo", "drone bay", "fighter bay")):
                    continue
                item_input = row.get("item_name_en") or row.get("item_name_local")
                item_id, _label = _resolve_type_id(item_input)
                item = db.getItem(item_id) if item_id is not None else None
                if item is None:
                    errors.append({"type": "item_not_found", "name": item_input})
                    continue
                items.append(item)
            report = _skill_report(character, metadata, items)
            results.append(
                {
                    "fitting_id": fitting.get("id"),
                    "fitting_name": fitting.get("fitting_name"),
                    "ship": fitting.get("ship_name_local")
                    or fitting.get("ship_name_en"),
                    "errors": errors,
                    "skill_requirements": report,
                }
            )
        return _json_value(results)


def recommend_fittings_for_character(
    fittings: list[dict[str, Any]],
    character_ref: Any,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Run every candidate fitting through Eos with one character skill snapshot."""
    from eve_lightrag.eve_character_skills import get_character_skill_levels

    levels, metadata = get_character_skill_levels(character_ref)
    evaluated: list[dict[str, Any]] = []
    for fitting in fittings:
        payload = fitting_to_simulation_payload(fitting, "custom")
        payload["character"] = metadata.get("character_name", str(character_ref))
        payload["_skill_levels"] = levels
        payload["_skillpoints_by_id"] = metadata.get("skillpoints_by_id", {})
        try:
            simulation = simulate_fitting(payload)
            requirements = simulation.get("skill_requirements", {})
            skill_ready = (
                bool(simulation.get("valid"))
                and int(requirements.get("missing_count", 0) or 0) == 0
            )
            evaluated.append(
                {
                    "fitting_id": fitting.get("id"),
                    "fitting_name": fitting.get("fitting_name"),
                    "ship": fitting.get("ship_name_local")
                    or fitting.get("ship_name_en"),
                    "source_file": fitting.get("source_file"),
                    "valid": bool(simulation.get("valid")),
                    "skill_ready": skill_ready,
                    "errors": simulation.get("errors", []),
                    "skill_requirements": requirements,
                    "metrics": {
                        "cpu": (simulation.get("resources", {}).get("cpu", {}) or {}),
                        "powergrid": (
                            simulation.get("resources", {}).get("powergrid", {}) or {}
                        ),
                        "calibration": (
                            simulation.get("resources", {}).get("calibration", {}) or {}
                        ),
                        "dps": (
                            simulation.get("damage", {}).get("dps", {}) or {}
                        ).get("total"),
                        "volley": (
                            simulation.get("damage", {}).get("volley", {}) or {}
                        ).get("total"),
                        "ehp": (
                            simulation.get("ship_stats", {}) or {}
                        ).get("ehp"),
                        "capacitor_stable": (
                            simulation.get("ship_stats", {}) or {}
                        ).get("capacitor_stable"),
                        "capacitor_used_per_second": (
                            simulation.get("ship_stats", {}) or {}
                        ).get("capacitor_used_per_second"),
                        "tank": simulation.get("tank", {}) or {},
                    },
                    "simulation": simulation,
                }
            )
        except Exception as exc:
            evaluated.append(
                {
                    "fitting_id": fitting.get("id"),
                    "fitting_name": fitting.get("fitting_name"),
                    "ship": fitting.get("ship_name_local")
                    or fitting.get("ship_name_en"),
                    "source_file": fitting.get("source_file"),
                    "valid": False,
                    "skill_ready": False,
                    "errors": [{"type": "simulation_error", "error": str(exc)}],
                }
            )

    def rank(row: dict[str, Any]) -> tuple[int, int, float]:
        report = row.get("skill_requirements", {}) or {}
        dps = row.get("metrics", {}).get("dps")
        try:
            dps_value = float(dps or 0)
        except (TypeError, ValueError):
            dps_value = 0.0
        return (
            0 if row.get("skill_ready") else (1 if row.get("valid") else 2),
            int(report.get("missing_sp_total", 10**18) or 0),
            -dps_value,
        )

    evaluated.sort(key=rank)
    return _json_value(evaluated[: max(1, int(max_results))])


def simulate_fitting(payload: dict[str, Any]) -> dict[str, Any]:
    """Simulate one fit and return JSON-safe metrics and validation errors."""
    if not isinstance(payload, dict):
        raise ValueError("装配模拟输入必须是 JSON 对象")
    ship_input = payload.get("ship", payload.get("ship_name", payload.get("ship_id")))
    if ship_input is None:
        raise ValueError("缺少 ship / ship_name / ship_id")

    slot_keys = ("high", "mid", "low", "rig", "subsystem", "service")
    module_entries = {
        slot: _normalize_module_entries(payload.get(slot, [])) for slot in slot_keys
    }
    # Accept the common compact shape: {"modules": [{"name": "..."}]}.
    for entry in payload.get("modules", []) or []:
        if not isinstance(entry, dict):
            raise ValueError("modules 中的装备必须是对象")
        slot = str(entry.get("slot", "")).casefold()
        if slot not in slot_keys:
            raise ValueError(f"modules 装备缺少有效 slot：{slot}")
        module_entries[slot].extend(_normalize_module_entries([entry]))

    with _EOS_LOCK:
        db, fit_module, ship_module, module_module = _ensure_eos()
        character_module = importlib.import_module("eos.saveddata.character")
        ship_id, ship_label = _resolve_type_id(ship_input)
        if ship_id is None:
            raise ValueError(f"没有找到舰船：{ship_input}")
        ship_item = db.getItem(ship_id)
        if ship_item is None:
            raise ValueError(f"Pyfa 数据中没有舰船 typeID：{ship_id}")

        fit = fit_module.Fit(ship=ship_module.Ship(ship_item), name="LightRAG simulation")
        character, skill_metadata = _build_character(payload, character_module)
        if character is not None:
            fit.character = character
        errors: list[dict[str, Any]] = []
        resolved_modules: list[dict[str, Any]] = []
        added_modules: list[tuple[Any, dict[str, Any]]] = []
        used_charges: list[dict[str, Any]] = []
        available_charges_by_module: dict[int, dict[str, Any]] = {}

        for slot, entries in module_entries.items():
            for entry in entries:
                type_id, label = _resolve_type_id(entry["name"])
                if type_id is None:
                    errors.append({"type": "not_found", "slot": slot, "name": label})
                    continue
                item = db.getItem(type_id)
                if item is None:
                    errors.append(
                        {"type": "not_found", "slot": slot, "name": label, "type_id": type_id}
                    )
                    continue
                try:
                    module = module_module.Module(item)
                    module.owner = fit
                    fit.modules.append(module)
                    if module.isInvalid:
                        errors.append(
                            {
                                "type": "invalid_module",
                                "slot": slot,
                                "name": label,
                                "type_id": type_id,
                            }
                        )
                        continue
                    resolved_modules.append(
                        {
                            "slot_requested": slot,
                            "slot_actual": int(module.slot),
                            "name": item.name,
                            "type_id": type_id,
                            "state": entry["state"],
                        }
                    )
                    added_modules.append((module, entry))
                    damage_charges = _damage_charge_candidates(module)
                    if damage_charges and type_id not in available_charges_by_module:
                        seen_charge_ids: set[int] = set()
                        unique_charges = []
                        for charge in damage_charges:
                            charge_id_int = int(charge.ID)
                            if charge_id_int in seen_charge_ids:
                                continue
                            seen_charge_ids.add(charge_id_int)
                            unique_charges.append(_charge_summary(charge))
                        available_charges_by_module[type_id] = {
                            "module": item.name,
                            "module_type_id": type_id,
                            "charges": unique_charges,
                        }
                    charge_name = entry.get("charge")
                    if charge_name is not None:
                        charge_id, charge_label = _resolve_type_id(charge_name)
                        charge_item = db.getItem(charge_id) if charge_id is not None else None
                        if charge_item is None or not module.isValidCharge(charge_item):
                            errors.append(
                                {
                                    "type": "invalid_charge",
                                    "module": label,
                                    "charge": charge_label,
                                }
                            )
                        else:
                            module.charge = charge_item
                            used_charges.append(
                                {
                                    "module": item.name,
                                    "module_type_id": type_id,
                                    "charge": _charge_summary(charge_item),
                                    "source": "explicit",
                                }
                            )
                    elif damage_charges:
                        charge_item = damage_charges[0]
                        module.charge = charge_item
                        used_charges.append(
                            {
                                "module": item.name,
                                "module_type_id": type_id,
                                "charge": _charge_summary(charge_item),
                                "source": "auto_highest_damage",
                            }
                        )
                except Exception as exc:
                    errors.append(
                        {
                            "type": "module_error",
                            "slot": slot,
                            "name": label,
                            "type_id": type_id,
                            "error": str(exc),
                        }
                    )

        for module, entry in added_modules:
            state_map = {
                "offline": -1,
                "online": 0,
                "active": 1,
                "overheated": 2,
            }
            try:
                module.state = state_map.get(entry["state"], 0)
            except Exception:
                errors.append(
                    {
                        "type": "invalid_state",
                        "name": module.item.name,
                        "state": entry["state"],
                    }
                )

        fit.calculateModifiedAttributes()
        for module in fit.modules:
            if module.isEmpty:
                continue
            if not module.fits(fit):
                errors.append(
                    {
                        "type": "fit_restriction",
                        "name": module.item.name,
                        "type_id": module.item.ID,
                        "reason": "槽位、装备组上限、硬点或舰船适配限制不满足",
                    }
                    )

        cpu_total = _metric(fit, "cpuOutput")
        powergrid_total = _metric(fit, "powerOutput")
        calibration_total = _metric(fit, "upgradeCapacity")
        resource_errors: list[dict[str, Any]] = []
        for resource, used, total in (
            ("cpu", fit.cpuUsed, cpu_total),
            ("powergrid", fit.pgUsed, powergrid_total),
            ("calibration", fit.calibrationUsed, calibration_total),
        ):
            if used > total:
                resource_errors.append(
                    {
                        "type": f"{resource}_overloaded",
                        "used": _json_value(used),
                        "total": _json_value(total),
                        "remaining": _json_value(total - used),
                    }
                )
        errors.extend(resource_errors)
        dps = fit.getTotalDps()
        volley = fit.getTotalVolley()
        ehp = fit.ehp
        tank = fit.tank
        sustainable_tank = fit.sustainableTank
        used_charge_groups: dict[tuple[int, int], dict[str, Any]] = {}
        for row in used_charges:
            charge = row.get("charge", {}) or {}
            key = (
                int(row.get("module_type_id", 0) or 0),
                int(charge.get("type_id", 0) or 0),
            )
            grouped = used_charge_groups.setdefault(
                key,
                {
                    "module": row.get("module"),
                    "module_type_id": row.get("module_type_id"),
                    "charge": charge,
                    "source": row.get("source"),
                    "count": 0,
                },
            )
            grouped["count"] += 1
        result = {
            "skill": "pyfa_fitting_simulator",
            "engine": "Pyfa Eos",
            "ship": {
                "input": ship_input,
                "name": fit.ship.name,
                "type_id": fit.ship.item.ID,
            },
            "valid": not errors and bool(fit.fits),
            "errors": errors,
            "modules": resolved_modules,
            "charges": {
                "used": used_charges,
                "used_by_module": list(used_charge_groups.values()),
                "available_by_module": list(available_charges_by_module.values()),
            },
            "resources": {
                "cpu": {
                    "used": fit.cpuUsed,
                    "total": cpu_total,
                    "remaining": _json_value(cpu_total - fit.cpuUsed),
                },
                "powergrid": {
                    "used": fit.pgUsed,
                    "total": powergrid_total,
                    "remaining": _json_value(powergrid_total - fit.pgUsed),
                },
                "calibration": {
                    "used": fit.calibrationUsed,
                    "total": calibration_total,
                    "remaining": _json_value(calibration_total - fit.calibrationUsed),
                },
            },
            "slots": {
                "high": {"used": fit.getSlotsUsed(3), "total": _metric(fit, "hiSlots")},
                "mid": {"used": fit.getSlotsUsed(2), "total": _metric(fit, "medSlots")},
                "low": {"used": fit.getSlotsUsed(1), "total": _metric(fit, "lowSlots")},
                "rig": {"used": fit.getSlotsUsed(4), "total": _metric(fit, "rigSlots")},
            },
            "ship_stats": {
                "shield_capacity": _metric(fit, "shieldCapacity"),
                "armor_hp": _metric(fit, "armorHP"),
                "hull_hp": _metric(fit, "hp"),
                "max_velocity": _json_value(fit.maxSpeed),
                "signature_radius": _metric(fit, "signatureRadius"),
                "align_time": _json_value(fit.alignTime),
                "ehp": _json_value(ehp),
                "capacitor_stable": bool(fit.capStable),
                "capacitor_state": _json_value(fit.capState),
                "capacitor_used_per_second": _json_value(fit.capUsed),
                "capacitor_recharge_per_second": _json_value(fit.capRecharge),
            },
            "damage": {
                "dps": _damage_map(dps),
                "volley": _damage_map(volley),
            },
            "tank": {
                "passive_shield_recharge": _json_value(
                    tank.get("passiveShield", 0)
                ),
                "shield_repair": _json_value(tank.get("shieldRepair", 0)),
                "armor_repair": _json_value(tank.get("armorRepair", 0)),
                "hull_repair": _json_value(tank.get("hullRepair", 0)),
                "sustainable_passive_shield_recharge": _json_value(
                    sustainable_tank.get("passiveShield", 0)
                ),
                "sustainable_shield_repair": _json_value(
                    sustainable_tank.get("shieldRepair", 0)
                ),
                "sustainable_total_shield": _json_value(
                    (sustainable_tank.get("passiveShield", 0) or 0)
                    + (sustainable_tank.get("shieldRepair", 0) or 0)
                ),
            },
            "source": {
                "repository": "https://github.com/pyfa-org/Pyfa",
                "local_path": str(PYFA_ROOT),
                "license": "Pyfa GPL-3.0-or-later; embedded Eos LGPL-2.1",
            },
            "skills": skill_metadata,
        }
        if character is not None:
            result["skill_requirements"] = _skill_report(
                character,
                skill_metadata,
                [ship_item] + [module.item for module, _entry in added_modules],
            )
        return _json_value(result)


def simulate_fitting_json(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    return simulate_fitting(payload)
