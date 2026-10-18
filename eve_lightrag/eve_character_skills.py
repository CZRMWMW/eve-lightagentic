"""ESI skills using the dedicated authorization store, never legacy tables."""

from __future__ import annotations

from typing import Any


def _services() -> tuple[Any, Any, Any]:
    from eve_lightrag import eve_esi_bindings
    return eve_esi_bindings, eve_esi_bindings, eve_esi_bindings


def list_authorized_characters() -> list[dict[str, Any]]:
    """Return only the public identity fields of authorized characters."""
    _esi_auth, _skill_service, db = _services()
    return [
        {
            "character_id": int(row["character_id"]),
            "character_name": str(row["character_name"]),
        }
        for row in db.list_all_characters()
    ]


def resolve_character(character: Any = None) -> dict[str, Any]:
    """Resolve a character ID or name from the dedicated authorization table."""
    characters = list_authorized_characters()
    if character is None or str(character).strip() == "":
        if len(characters) == 1:
            return characters[0]
        raise ValueError("已授权多个角色，请提供 character_id 或角色名")

    text = str(character).strip()
    if text.isdigit():
        character_id = int(text)
        for item in characters:
            if item["character_id"] == character_id:
                return item
        raise ValueError(f"角色 {character_id} 不在已授权角色列表中")

    normalized = text.casefold()
    exact = [
        item
        for item in characters
        if item["character_name"].casefold() == normalized
    ]
    if len(exact) == 1:
        return exact[0]
    partial = [
        item
        for item in characters
        if normalized in item["character_name"].casefold()
    ]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise ValueError(f"没有找到已授权角色：{character}")
    raise ValueError(f"角色名匹配到多个角色：{character}")


def find_character_in_text(text: str) -> dict[str, Any] | None:
    """Find an authorized character name embedded in a natural-language query."""
    normalized = str(text or "").casefold()
    matches = [
        item
        for item in list_authorized_characters()
        if item["character_name"].casefold() in normalized
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _skill_name_map() -> dict[int, str]:
    """Use the local Pyfa SDE for stable English skill names."""
    try:
        from eve_lightrag.pyfa_skill import _ensure_eos

        _ensure_eos()
        from eos.saveddata.character import Character

        return {
            int(skill_id): str(item.typeName)
            for skill_id, item in Character.getSkillIDMap().items()
        }
    except Exception:
        return {}


def get_character_skill_data(
    character: Any = None,
    include_queue: bool = True,
    include_attributes: bool = False,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Fetch a character's current ESI skills without exposing credentials."""
    resolved = resolve_character(character)
    esi_auth, skill_service, _db = _services()
    token = esi_auth.get_valid_access_token(resolved["character_id"])
    raw = skill_service.fetch_character_skills(
        resolved["character_id"], token, force_refresh
    )
    names = _skill_name_map()
    trained = [
        {
            "skill_id": int(item["skill_id"]),
            "name": names.get(int(item["skill_id"])),
            "active_level": int(item.get("active_skill_level", 0)),
            "trained_level": int(item.get("trained_skill_level", 0)),
            "skillpoints": int(item.get("skillpoints_in_skill", 0)),
        }
        for item in raw.get("skills", [])
    ]
    result: dict[str, Any] = {
        "skill": "eve_character_skills_esi",
        "character": resolved,
        "total_sp": int(raw.get("total_sp", 0)),
        "unallocated_sp": int(raw.get("unallocated_sp", 0)),
        "trained_count": len(trained),
        "skills": trained,
        "skillpoints_by_id": {
            str(item["skill_id"]): int(item.get("skillpoints_in_skill", 0))
            for item in raw.get("skills", [])
        },
    }
    if include_queue:
        queue = skill_service.fetch_skill_queue(
            resolved["character_id"], token, force_refresh
        )
        result["skill_queue"] = [
            {
                "queue_position": int(item.get("queue_position", 0)),
                "skill_id": int(item["skill_id"]),
                "name": names.get(int(item["skill_id"])),
                "target_level": int(item.get("finished_level", 0)),
                "start_date": item.get("start_date"),
                "finish_date": item.get("finish_date"),
            }
            for item in sorted(queue, key=lambda row: row.get("queue_position", 0))
        ]
    if include_attributes:
        result["attributes"] = skill_service.fetch_character_attributes(
            resolved["character_id"], token, force_refresh
        )
    return result


def get_character_skill_levels(
    character: Any = None,
    force_refresh: bool = False,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Return ``{skill_type_id: active_level}`` plus safe character metadata."""
    data = get_character_skill_data(
        character,
        include_queue=False,
        include_attributes=False,
        force_refresh=force_refresh,
    )
    levels = {
        int(skill["skill_id"]): int(skill["active_level"])
        for skill in data["skills"]
    }
    return levels, {
        "character_id": data["character"]["character_id"],
        "character_name": data["character"]["character_name"],
        "loaded_count": len(levels),
        "total_sp": data["total_sp"],
        "skillpoints_by_id": data.get("skillpoints_by_id", {}),
    }
