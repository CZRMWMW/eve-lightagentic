"""Official SDE name pairs; identifiers remain independent of display language."""

import re
import threading
import time
from collections import defaultdict

from eve_lightrag.eve_static_data import connection

TABLE = "eve_sde_names"
_lock = threading.RLock()
_cache = None
_loaded = 0.0
_protected = re.compile(r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)|^\[[^\]\n]+,[^\]\n]+\]\n[\s\S]*\Z|`[^`\n]*`|https?://[^\s<>]+|\]\([^\n)]*\)|\bPyfa(?:/Eos)?\b", re.MULTILINE | re.IGNORECASE)
# These are also SDE names, but ordinary prose cannot disambiguate them.
# Exact lookups remain available with dataset/ID, even for these terms.
_ambiguous_prose = frozenset(('static', 'test', 'access', 'tool', 'tools',
    'online', 'active', 'ship', 'structure', 'module', 'skill', 'skills',
    'small', 'medium', 'large', 'character', 'group', 'type', 'none'))


def refresh_names():
    """Materialize every named SDE entity without inventing translations."""
    global _cache
    with _lock, connection() as conn, conn.cursor() as cursor:
        cursor.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
            dataset VARCHAR(96) NOT NULL,
            entity_id VARCHAR(128) NOT NULL,
            name_en VARCHAR(512) NOT NULL,
            name_zh VARCHAR(512) NOT NULL,
            PRIMARY KEY(dataset, entity_id),
            KEY idx_name_en(name_en(191)), KEY idx_name_zh(name_zh(191))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""")
        cursor.execute(f"""INSERT INTO {TABLE} (dataset,entity_id,name_en,name_zh)
            SELECT dataset,record_key,name_en,name_zh FROM eve_sde_records
            WHERE name_en<>'' OR name_zh<>''
            ON DUPLICATE KEY UPDATE name_en=VALUES(name_en),name_zh=VALUES(name_zh)""")
        cursor.execute(f"""DELETE n FROM {TABLE} n LEFT JOIN eve_sde_records s
            ON n.dataset=s.dataset AND n.entity_id=s.record_key
            WHERE s.record_key IS NULL OR (s.name_en='' AND s.name_zh='')""")
        conn.commit()
        cursor.execute(f"SELECT dataset,COUNT(*) AS count FROM {TABLE} GROUP BY dataset")
        counts = cursor.fetchall()
        _cache = None
        return counts


def _index():
    global _cache, _loaded
    with _lock:
        if _cache is not None and time.monotonic() - _loaded < 300:
            return _cache
        with connection() as conn, conn.cursor() as cursor:
            cursor.execute(f"SELECT dataset,entity_id,name_en,name_zh FROM {TABLE}")
            rows = cursor.fetchall()
        by_name = defaultdict(list)
        for row in rows:
            for name in {row['name_en'], row['name_zh']} - {''}:
                by_name[name.casefold()].append(row)
        # A trie avoids compiling tens of thousands of names into one regex.
        trie = {}
        for name in by_name:
            if name in _ambiguous_prose or len(name) < 2 or (name.isascii() and len(name) < 3):
                continue
            node = trie
            for char in name:
                node = node.setdefault(char, {})
            node[None] = name
        _cache = by_name, trie
        _loaded = time.monotonic()
        return _cache


def resolve_name(name, dataset=None):
    """Exact bidirectional lookup; never choose arbitrarily among homonyms."""
    rows = _index()[0].get(str(name).strip().casefold(), [])
    rows = [r for r in rows if dataset is None or r['dataset'] == dataset]
    return dict(rows[0]) if len(rows) == 1 else None


def _matches(text):
    by_name, trie = _index()
    # ASCII folding preserves string offsets for arbitrary Chinese input.
    folded = text.translate(str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'))
    pos = 0
    while pos < len(text):
        node, end, best = trie, pos, None
        while end < len(text) and folded[end] in node:
            node = node[folded[end]]
            end += 1
            if None in node:
                name = node[None]
                left_ok = not (name[0].isascii() and name[0].isalnum() and pos and (text[pos-1].isascii() and (text[pos-1].isalnum() or text[pos-1] == '_')))
                right_ok = not (name[-1].isascii() and name[-1].isalnum() and end < len(text) and text[end].isascii() and (text[end].isalnum() or text[end] == '_'))
                if left_ok and right_ok:
                    best = end, by_name[name]
        if best:
            yield pos, best[0], best[1]
            pos = best[0]
        else:
            pos += 1


def name_context(text):
    pairs = {}
    for _, _, rows in _matches(text):
        for row in rows:
            if row['name_zh'] and row['name_en']:
                pairs[(row['dataset'], row['entity_id'])] = row
    if not pairs:
        return ''
    import json
    return ('\n【SDE 官方名称对照】\n专有名词必须使用下列官方中文名，禁止自行直译。'
            '内部查询以实体ID为准；Pyfa/EFT输入使用英文名或typeID。'
            '没有中文映射的保留英文；不要翻译代码、EFT、URL、角色名。\n'
            + json.dumps(list(pairs.values()), ensure_ascii=False, separators=(',', ':')))


def localize_text(text):
    """Translate unambiguous prose names, preserving EFT/code/links verbatim."""
    def replace(part):
        pieces, previous = [], 0
        for start, end, rows in _matches(part):
            if start < previous:
                continue
            names = {r['name_zh'] for r in rows}
            replacement = next(iter(names)) if len(names) == 1 and '' not in names else part[start:end]
            # An explicitly paired English ship name establishes identity;
            # discard its generated Chinese transliteration, not just English.
            if (replacement != part[start:end] and part[start:end].isascii()
                    and all(r['dataset'] == 'types' for r in rows)
                    and replacement.endswith('级')):
                paired = re.match(r'(\*\*|__)?[ \t]*[（(][\u3400-\u9fff/／、· \-]{1,40}级[）)]', part[end:])
                if paired:
                    replacement += paired[1] or ''
                    end += paired.end()
            pieces.extend((part[previous:start], replacement))
            previous = end
        return ''.join(pieces) + part[previous:]
    parts, previous = [], 0
    for match in _protected.finditer(text):
        parts.extend((replace(text[previous:match.start()]), match.group()))
        previous = match.end()
    return ''.join(parts) + replace(text[previous:])


class NameStream:
    """Buffer lines so names split over LLM chunks still match."""
    def __init__(self):
        self.pending = ''
        self.fence = None
        self.eft = False

    def feed(self, chunk='', final=False):
        self.pending += chunk
        output = []
        while '\n' in self.pending or (final and self.pending):
            if '\n' in self.pending:
                line, self.pending = self.pending.split('\n', 1)
                line += '\n'
            else:
                line, self.pending = self.pending, ''
            marker = re.match(r'^\s*(`{3,}|~{3,})', line)
            if self.eft:
                output.append(line)
            elif self.fence:
                output.append(line)
                if marker and marker[1][0] == self.fence[0] and len(marker[1]) >= len(self.fence):
                    self.fence = None
            elif marker:
                self.fence = marker[1]
                output.append(line)
            elif re.match(r'^\[[^\]\n]+,[^\]\n]+\]\n', line):
                self.eft = True
                output.append(line)
            else:
                output.append(localize_text(line))
        return ''.join(output)
