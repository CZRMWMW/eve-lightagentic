# EVE LightAgentic

面向 EVE Online 的知识检索与装配计算服务，基于 LightRAG 1.5.7，结合知识库、静态数据、角色技能和 Pyfa Eos 计算引擎。

## 功能

- Neo4j 知识图谱与 Milvus 文档检索。
- MySQL 舰船装配、SDE 静态数据查询和中英文名称匹配。
- 根据问题选择查询工具，组合知识检索、角色技能查询与装配计算。
- Pyfa Eos 计算 CPU、能量栅格、电容、伤害与防御属性，支持角色技能、全 V 和全 0 技能方案。
- 返回计算使用的弹药，以及去重后的兼容弹药列表。
- 独立 ESI SSO 角色绑定页面，加密保存授权令牌。
- 同步、流式和排队查询接口。

本仓库只提供程序。知识库原文、装配文件、SDE 数据、角色授权记录和生产配置不随代码分发。

## 安装

建议使用 Python 3.12，在独立虚拟环境中安装，避免覆盖其他 LightRAG 项目。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[api]'
cp .env.example .env
```

填写 `.env` 中的模型、数据库和鉴权配置。Neo4j、Milvus 和 MySQL 需要单独部署；模型与向量模型必须与导入知识库时使用的配置兼容。更多模型和存储配置参见 [LightRAG 文档](https://github.com/HKUDS/LightRAG)。

```bash
lightrag-server --host 127.0.0.1 --port 9621
```

默认建议只监听本机，通过 HTTPS 反向代理对外提供服务。安装时沿用上游发行包名称 `lightrag-hku`，服务入口保持兼容。

## 数据初始化

先创建 MySQL 数据库并授予应用账号所需权限。将合法取得的 SDE JSONL 文件放入 `EVE_SDE_DIR`，装配 XML 文件放入 `EVE_FITTINGS_SOURCE_DIR`。

```bash
python -m eve_lightrag.manage_data sde
python -m eve_lightrag.manage_data fittings
python -m eve_lightrag.manage_data esi-init
python -m eve_lightrag.manage_data esi-configure
```

ESI 配置命令交互读取 Client ID、Client Secret 和 HTTPS 回调地址，Secret 输入不回显。回调地址应为 `https://你的域名/eve/oauth/callback`，申请技能与技能队列读取权限。绑定页面为 `/eve/bind`。加密密钥位于 `EVE_DATA_DIR`，应与数据库共同备份，且不得提交到 Git。

## Pyfa

单独取得 [Pyfa](https://github.com/pyfa-org/Pyfa) 源码及其配套数据，安装相应依赖，将 `PYFA_ROOT` 指向该目录。适配层动态加载 Eos，不分发 Pyfa 源码、个人装配或缓存。实际部署需核对 Pyfa 版本、数据库版本及 Eos 依赖兼容性。

装配计算支持既有装配和 EFT 文本。自动弹药选择按兼容弹药的基础伤害排序，并不代表对任意目标的最高实际 DPS。无人机和植入体尚未完整接入；装配资源合法与角色满足技能门槛是两个检查维度。

## 接口

```bash
curl http://127.0.0.1:9621/query \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_API_KEY' \
  -d '{"query":"查询洛基的装配方案","mode":"hybrid"}'
```

| 接口 | 用途 |
| --- | --- |
| `POST /query` | 等待完整查询结果 |
| `POST /query/stream` | 流式查询 |
| `POST /query/queued` | 提交排队任务，取得 job_id |
| `GET /query/queued/{job_id}` | 轮询任务状态与结果 |

请求和响应的完整字段以运行服务的 `/docs` 为准。机器人应保留服务返回的技能来源、模拟结果和工具信息，避免二次生成时改写计算数值或翻译官方名称。

队列位于单进程内存中，不使用 Redis，不持久化；重启会丢失任务。多进程部署时，提交与轮询必须落到同一进程，或另行实现共享任务存储。客户端超时和轮询策略应覆盖模型排队时间。

## 开发与验证

```bash
python -m unittest discover -s tests
pip wheel --no-deps --no-build-isolation . -w dist
```

离线测试不访问生产数据库或 ESI。真实查询仍需配置模型、数据和 Pyfa 环境后进行集成验证。调试请求记录默认关闭，开启后可能记录问题与角色标识，应限制访问并设置保留期限。

## 许可

LightRAG 部分保留上游 MIT 许可与版权，见 [LICENSE](LICENSE)。扩展代码按同一许可提供。第三方组件及 EVE 数据权利说明见 [NOTICE.md](NOTICE.md)。
