"""Dedicated ESI authorization storage for LightRAG skills."""

import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from cryptography.fernet import Fernet
import jwt
import pymysql
import requests

from eve_lightrag.eve_static_data import connection

SCOPES = ('esi-skills.read_skills.v1', 'esi-skills.read_skillqueue.v1')
TOKEN_URL = 'https://login.eveonline.com/v2/oauth/token'
AUTHORIZE_URL = 'https://login.eveonline.com/v2/oauth/authorize/'
_cache = {}
_cache_lock = threading.Lock()
_jwk = jwt.PyJWKClient('https://login.eveonline.com/oauth/jwks', timeout=15)


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ReauthorizationRequired(ValueError):
    pass


def _cipher():
    path = Path(os.getenv('EVE_SKILL_KEY_FILE', str(Path(os.getenv('EVE_DATA_DIR', 'data')) / '.eve_skill_tokens.key')))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(Fernet.generate_key())
    return Fernet(path.read_bytes().strip())


def _encrypt(value):
    return _cipher().encrypt(value.encode()).decode()


def _decrypt(value):
    return _cipher().decrypt(value.encode()).decode()


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def ensure_schema():
    with connection() as conn, conn.cursor() as q:
        q.execute('''CREATE TABLE IF NOT EXISTS eve_skill_oauth_clients (
            client_key VARCHAR(64) PRIMARY KEY, client_id VARCHAR(128) NOT NULL,
            secret_cipher TEXT NOT NULL, callback_url VARCHAR(1024) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4''')
        q.execute('''CREATE TABLE IF NOT EXISTS eve_skill_characters (
            character_id BIGINT UNSIGNED PRIMARY KEY,
            character_name VARCHAR(128) NOT NULL,
            owner_hash VARCHAR(255) NOT NULL DEFAULT '',
            client_key VARCHAR(64) NOT NULL,
            access_cipher TEXT NOT NULL, refresh_cipher TEXT NOT NULL,
            scopes TEXT NOT NULL, token_expires_at DATETIME NOT NULL,
            status VARCHAR(32) NOT NULL DEFAULT 'active',
            migrated_from VARCHAR(255) NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            KEY idx_character_name(character_name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4''')
        q.execute('''CREATE TABLE IF NOT EXISTS eve_skill_bindings (
            platform VARCHAR(32) NOT NULL, user_id VARCHAR(128) NOT NULL,
            character_id BIGINT UNSIGNED NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(platform,user_id,character_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4''')
        q.execute('''CREATE TABLE IF NOT EXISTS eve_skill_oauth_states (
            state_hash CHAR(64) PRIMARY KEY, browser_hash CHAR(64) NOT NULL,
            verifier_cipher TEXT NOT NULL, platform VARCHAR(32) NULL,
            user_id VARCHAR(128) NULL, expires_at DATETIME NOT NULL,
            KEY idx_state_expiry(expires_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4''')
        q.execute('''CREATE TABLE IF NOT EXISTS eve_skill_bind_links (
            ticket_hash CHAR(64) PRIMARY KEY, platform VARCHAR(32) NOT NULL,
            user_id VARCHAR(128) NOT NULL, expires_at DATETIME NOT NULL,
            KEY idx_link_expiry(expires_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4''')
        conn.commit()


def configure_client(client_id, client_secret, callback_url, client_key='binding'):
    if not callback_url.startswith('https://'):
        raise ValueError('ESI 回调地址必须使用 HTTPS')
    with connection() as conn, conn.cursor() as q:
        q.execute('''INSERT INTO eve_skill_oauth_clients VALUES (%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE client_id=VALUES(client_id),
            secret_cipher=VALUES(secret_cipher),callback_url=VALUES(callback_url)''',
            (client_key, client_id, _encrypt(client_secret), callback_url))
        conn.commit()


def _client(key='binding'):
    with connection() as conn, conn.cursor() as q:
        q.execute('SELECT * FROM eve_skill_oauth_clients WHERE client_key=%s', (key,))
        row = q.fetchone()
    if not row:
        raise ValueError('专用 ESI 应用尚未配置 Client ID、Client Secret 和 HTTPS 回调地址')
    return row


def binding_ready():
    try:
        _client()
        return True
    except ValueError:
        return False




def list_all_characters():
    with connection() as conn, conn.cursor() as q:
        q.execute("SELECT character_id,character_name FROM eve_skill_characters WHERE status='active' ORDER BY character_name")
        return q.fetchall()


def create_bind_link(platform, user_id):
    ticket = secrets.token_urlsafe(32)
    with connection() as conn, conn.cursor() as q:
        q.execute('DELETE FROM eve_skill_bind_links WHERE expires_at<UTC_TIMESTAMP()')
        q.execute('INSERT INTO eve_skill_bind_links VALUES (%s,%s,%s,%s)',
                  (_hash(ticket),platform,user_id,_utcnow()+timedelta(minutes=15)))
        conn.commit()
    return ticket


def begin_authorization(ticket=None):
    client = _client()
    state, browser, verifier = (secrets.token_urlsafe(32) for _ in range(3))
    platform = user_id = None
    with connection() as conn, conn.cursor() as q:
        if ticket:
            q.execute('SELECT * FROM eve_skill_bind_links WHERE ticket_hash=%s AND expires_at>UTC_TIMESTAMP() FOR UPDATE', (_hash(ticket),))
            link = q.fetchone()
            if not link:
                raise ValueError('绑定链接无效或已过期，请重新获取')
            platform,user_id = link['platform'],link['user_id']
            q.execute('DELETE FROM eve_skill_bind_links WHERE ticket_hash=%s', (_hash(ticket),))
        q.execute('DELETE FROM eve_skill_oauth_states WHERE expires_at<UTC_TIMESTAMP()')
        q.execute('INSERT INTO eve_skill_oauth_states VALUES (%s,%s,%s,%s,%s,%s)',
            (_hash(state),_hash(browser),_encrypt(verifier),platform,user_id,
             _utcnow()+timedelta(minutes=10)))
        conn.commit()
    import base64
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
    url = AUTHORIZE_URL + '?' + urlencode({'response_type':'code',
        'client_id':client['client_id'],'redirect_uri':client['callback_url'],
        'scope':' '.join(SCOPES),'state':state,
        'code_challenge':challenge,'code_challenge_method':'S256'})
    return url,browser


def complete_authorization(code,state,browser):
    with connection() as conn, conn.cursor() as q:
        q.execute('SELECT * FROM eve_skill_oauth_states WHERE state_hash=%s AND expires_at>UTC_TIMESTAMP() FOR UPDATE', (_hash(state),))
        entry = q.fetchone()
        if not entry or not secrets.compare_digest(entry['browser_hash'], _hash(browser)):
            raise ValueError('授权会话无效或过期，请从绑定页重新开始')
        q.execute('DELETE FROM eve_skill_oauth_states WHERE state_hash=%s', (_hash(state),))
        conn.commit()
    client = _client()
    tokens = _request_tokens(client,{'grant_type':'authorization_code','code':code,
        'redirect_uri':client['callback_url'],'code_verifier':_decrypt(entry['verifier_cipher'])})
    key = _jwk.get_signing_key_from_jwt(tokens['access_token'])
    claims = jwt.decode(tokens['access_token'],key.key,algorithms=['RS256'],
        audience='EVE Online',issuer=['login.eveonline.com','https://login.eveonline.com'])
    if claims.get('azp') != client['client_id']:
        raise ValueError('授权应用不匹配')
    subject = re.fullmatch(r'CHARACTER:EVE:(\d+)',claims.get('sub',''))
    if not subject:
        raise ValueError('授权返回的角色身份无效')
    scopes = claims.get('scp',[])
    scopes = scopes.split() if isinstance(scopes,str) else scopes
    if not set(SCOPES).issubset(scopes):
        raise ValueError('缺少技能读取权限，请重新授权')
    character_id = int(subject[1])
    with connection() as conn, conn.cursor() as q:
        q.execute('SELECT owner_hash FROM eve_skill_characters WHERE character_id=%s FOR UPDATE',(character_id,))
        previous = q.fetchone()
        if previous and previous['owner_hash'] and previous['owner_hash'] != claims.get('owner',''):
            q.execute('DELETE FROM eve_skill_bindings WHERE character_id=%s',(character_id,))
        q.execute('''INSERT INTO eve_skill_characters
            (character_id,character_name,owner_hash,client_key,access_cipher,
             refresh_cipher,scopes,token_expires_at,status)
            VALUES (%s,%s,%s,'binding',%s,%s,%s,%s,'active')
            ON DUPLICATE KEY UPDATE character_name=VALUES(character_name),
             owner_hash=VALUES(owner_hash),client_key=VALUES(client_key),
             access_cipher=VALUES(access_cipher),refresh_cipher=VALUES(refresh_cipher),
             scopes=VALUES(scopes),token_expires_at=VALUES(token_expires_at),status='active' ''',
            (character_id,claims['name'],claims.get('owner',''),
             _encrypt(tokens['access_token']),_encrypt(tokens['refresh_token']),
             ' '.join(scopes),_utcnow()+timedelta(seconds=int(tokens['expires_in']))))
        if entry['platform'] and entry['user_id']:
            q.execute('INSERT IGNORE INTO eve_skill_bindings (platform,user_id,character_id) VALUES (%s,%s,%s)',
                (entry['platform'],entry['user_id'],character_id))
        conn.commit()
    with _cache_lock:
        for key in list(_cache):
            if key[0] == character_id:
                del _cache[key]
    return {'character_id':character_id,'character_name':claims['name'],
            'platform':entry['platform'],'user_id':entry['user_id']}


def _request_tokens(client,data):
    response = requests.post(TOKEN_URL,data=data,
        auth=(client['client_id'],_decrypt(client['secret_cipher'])),timeout=20)
    if response.status_code != 200:
        error = ''
        try:
            error = response.json().get('error','')
        except ValueError:
            pass
        if error == 'invalid_grant':
            raise ReauthorizationRequired('ESI 授权已失效，请重新绑定')
        raise ValueError('EVE SSO 暂时无法完成授权，请稍后重试')
    return response.json()


def get_valid_access_token(character_id):
    # Row locking serializes rotating refresh tokens across threads/processes.
    with connection() as conn, conn.cursor() as q:
        q.execute("SELECT * FROM eve_skill_characters WHERE character_id=%s AND status='active' FOR UPDATE", (character_id,))
        row = q.fetchone()
        if not row:
            raise ValueError('该角色尚未绑定，请先完成 ESI 授权')
        if row['token_expires_at'] > _utcnow()+timedelta(seconds=60):
            return _decrypt(row['access_cipher'])
        client = _client(row['client_key'])
        try:
            tokens = _request_tokens(client,{'grant_type':'refresh_token',
                'refresh_token':_decrypt(row['refresh_cipher'])})
        except ReauthorizationRequired:
            q.execute("UPDATE eve_skill_characters SET status='reauthorization_required' WHERE character_id=%s",(character_id,))
            conn.commit()
            raise
        refresh = tokens.get('refresh_token')
        q.execute('''UPDATE eve_skill_characters SET access_cipher=%s,
            refresh_cipher=%s,token_expires_at=%s WHERE character_id=%s''',
            (_encrypt(tokens['access_token']),_encrypt(refresh) if refresh else row['refresh_cipher'],
             _utcnow()+timedelta(seconds=int(tokens['expires_in'])),character_id))
        conn.commit()
        return tokens['access_token']


def _fetch(character_id,token,resource,ttl,force_refresh):
    key = (int(character_id),resource)
    with _cache_lock:
        cached = _cache.get(key)
        if cached and not force_refresh and cached[0]>time.monotonic():
            return cached[1]
    response = requests.get(f'https://esi.evetech.net/latest/characters/{int(character_id)}/{resource}/',
        headers={'Authorization':f'Bearer {token}'},timeout=20)
    if response.status_code in (401,403):
        raise ValueError('角色授权失效或缺少该项权限，请重新绑定')
    response.raise_for_status()
    data = response.json()
    with _cache_lock:
        if len(_cache)>1000:
            _cache.clear()
        _cache[key] = time.monotonic()+ttl,data
    return data


def fetch_character_skills(character_id,token,force_refresh=False):
    return _fetch(character_id,token,'skills',300,force_refresh)


def fetch_skill_queue(character_id,token,force_refresh=False):
    return _fetch(character_id,token,'skillqueue',60,force_refresh)


def fetch_character_attributes(character_id,token,force_refresh=False):
    return _fetch(character_id,token,'attributes',3600,force_refresh)
