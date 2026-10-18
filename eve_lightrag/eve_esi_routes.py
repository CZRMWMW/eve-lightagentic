"""Public SSO page/callback and authenticated bot binding management."""

import html
import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from eve_lightrag import eve_esi_bindings as store
from eve_lightrag.query_skills import run_eve_db_call

HEADERS = {'Cache-Control':'no-store','Referrer-Policy':'no-referrer',
           'X-Content-Type-Options':'nosniff',
           'Content-Security-Policy':"default-src 'none'; style-src 'unsafe-inline'; img-src https://web.ccpgamescdn.com https://images.evetech.net; form-action 'self' https://login.eveonline.com; frame-ancestors 'none'; base-uri 'none'"}
COOKIE = 'eve_skill_sso'


class _SSOLogFilter(logging.Filter):
    def filter(self, record):
        if isinstance(record.args,tuple) and len(record.args)>=3:
            args = list(record.args)
            if isinstance(args[2],str) and ('/eve/oauth/callback?' in args[2] or '/eve/bind?' in args[2]):
                args[2] = args[2].split('?',1)[0]
                record.args = tuple(args)
        return True


class BindLinkRequest(BaseModel):
    platform: str = Field(default='qq', pattern=r'^[a-z0-9_-]{1,32}$')
    user_id: str = Field(min_length=1,max_length=128,pattern=r'^\S+$')


def _page(content,status=200):
    template = Path(__file__).with_name('eve_bind.html').read_text(encoding='utf-8')
    return HTMLResponse(template.replace('<!-- CONTENT -->',content),status_code=status,headers=HEADERS)


def create_esi_routes(auth):
    access_log = logging.getLogger('uvicorn.access')
    if not any(isinstance(f,_SSOLogFilter) for f in access_log.filters):
        access_log.addFilter(_SSOLogFilter())
    router = APIRouter(tags=['EVE ESI binding'])

    @router.get('/eve/bind',response_class=HTMLResponse,include_in_schema=False)
    async def page(ticket: str | None = Query(None,max_length=128)):
        ready = await run_eve_db_call(store.binding_ready)
        if not ready:
            return _page('<h1>EVE 角色绑定</h1><p>绑定暂不可用，请联系管理员。</p>',503)
        hidden = f'<input type="hidden" name="ticket" value="{html.escape(ticket,quote=True)}">' if ticket else ''
        return _page('<h1>EVE 角色绑定</h1><p>授权读取角色技能与训练队列。</p>'
            '<form method="post" action="/eve/bind/start">'+hidden+
            '<button type="submit" aria-label="通过 EVE Online 登录">'
            '<img src="https://web.ccpgamescdn.com/eveonlineassets/developers/eve-sso-login-black-large.png" '
            'alt="通过 EVE Online 登录" width="270" height="45"></button></form>')

    @router.post('/eve/bind/start',include_in_schema=False)
    async def start(request: Request,ticket: Annotated[str | None,Form(max_length=128)]=None):
        try:
            url,browser = await run_eve_db_call(store.begin_authorization,ticket)
        except ValueError as exc:
            return _page('<h1>暂时无法绑定</h1><p>'+html.escape(str(exc))+'</p>',400)
        response = RedirectResponse(url,status_code=303,headers=HEADERS)
        response.set_cookie(COOKIE,browser,max_age=600,httponly=True,secure=True,samesite='lax',path='/eve')
        return response

    @router.get('/eve/oauth/callback',include_in_schema=False)
    async def callback(request: Request,code: str=Query('',max_length=2048),
                       state: str=Query('',max_length=128),error: str=Query('',max_length=128)):
        if error or not code or not state:
            return _page('<h1>授权未完成</h1><p>请返回绑定入口重试。</p>',400)
        browser = request.cookies.get(COOKIE,'')
        try:
            result = await run_eve_db_call(store.complete_authorization,code,state,browser)
        except ValueError as exc:
            return _page('<h1>绑定失败</h1><p>'+html.escape(str(exc))+'</p>',400)
        except Exception:
            return _page('<h1>绑定失败</h1><p>EVE 授权服务暂不可用，请稍后重试。</p>',502)
        name = html.escape(result['character_name'])
        portrait = f"https://images.evetech.net/characters/{int(result['character_id'])}/portrait?size=128"
        response = _page(f'<h1>绑定成功</h1><img class="portrait" src="{portrait}" alt="角色头像" width="96" height="96">'
            f'<p class="character">{name}</p><p>角色 ID：{int(result["character_id"])}</p>')
        response.delete_cookie(COOKIE,path='/eve',secure=True,httponly=True,samesite='lax')
        return response

    @router.get('/eve/bind/config',dependencies=[Depends(auth)])
    async def config():
        ready = await run_eve_db_call(store.binding_ready)
        client = await run_eve_db_call(store._client) if ready else None
        return {'ready':ready,'callback_url':client['callback_url'] if client else None,
                'scopes':list(store.SCOPES),'page_path':'/eve/bind'}

    @router.post('/eve/bind/links',dependencies=[Depends(auth)])
    async def link(payload: BindLinkRequest):
        if not await run_eve_db_call(store.binding_ready):
            raise HTTPException(503,'专用 ESI 应用尚未配置')
        ticket = await run_eve_db_call(store.create_bind_link,payload.platform,payload.user_id)
        return {'bind_path':'/eve/bind?ticket='+ticket,'expires_in':900}

    @router.get('/eve/bind/characters',dependencies=[Depends(auth)])
    async def characters(platform: str=Query(...,min_length=1,max_length=32),
                         user_id: str=Query(...,min_length=1,max_length=128)):
        def lookup():
            with store.connection() as conn,conn.cursor() as q:
                q.execute('''SELECT c.character_id,c.character_name,c.status
                    FROM eve_skill_bindings b JOIN eve_skill_characters c USING(character_id)
                    WHERE b.platform=%s AND b.user_id=%s''',(platform,user_id))
                return q.fetchall()
        return {'characters':await run_eve_db_call(lookup)}

    @router.delete('/eve/bind/characters/{character_id}',dependencies=[Depends(auth)])
    async def unbind(character_id: int):
        def revoke():
            with store.connection() as conn,conn.cursor() as q:
                q.execute("UPDATE eve_skill_characters SET status='revoked',access_cipher='',refresh_cipher='' WHERE character_id=%s",(character_id,))
                changed = q.rowcount
                q.execute('DELETE FROM eve_skill_bindings WHERE character_id=%s',(character_id,))
                conn.commit()
                return changed
        if not await run_eve_db_call(revoke):
            raise HTTPException(404,'没有找到该角色授权')
        return {'status':'unbound','character_id':character_id}

    return router
