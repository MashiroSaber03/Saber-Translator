# Saber Translator 公网运行与部署方案

## 1. 目标与固定边界

- 公网地址：`https://saber.mashirosaber.work`
- 部署机器：当前这台 Windows 电脑
- 公网入口：复用机器上已经运行的 Cloudflare Tunnel
- 应用监听：只监听 `127.0.0.1:5100`，不开放路由器端口，不直接暴露局域网地址
- 本地桌面模式继续使用原有数据和端口，公网模式使用独立数据目录
- 不限制用户数量
- 不限制书本数量
- 唯一产品容量限制是每个用户的持久化资产总量：默认 `2 GiB`
- 管理员可以修改所有用户统一使用的资产额度

“资产”是写入后端对象存储的上传文件和生成文件。用户名额、书本记录、章节记录不参与额度判断。单次 HTTP 请求仍保留必要的安全上限，这属于上传入口防护，不是用户数量或书本数量限制。

## 2. 不过度设计原则

本方案明确不引入以下内容：

- 不引入 Docker、Kubernetes、Nginx、Redis、Celery 或 PostgreSQL
- 不另购云服务器、对象存储或托管数据库
- 不实现套餐、付费、用户等级、书本数量配额或并行的多套额度系统
- 不实现邮箱系统、短信系统或第三方登录；密码找回使用一次性恢复码
- 不实现复杂风控平台；只保留登录失败限速、可选的邀请码注册、CSRF 和资源归属校验
- 不实现第二套密钥数据库；网友的 Provider 密钥保存在其浏览器，后端只在内存中临时使用
- 不实现插件沙箱、公开网页抓取和本机模型服务的公网开放；公网模式直接关闭这些高风险入口
- 不改变现有 Cloudflare Tunnel 基座，只新增一个 hostname 到本地服务的路由

当前实现经过一次“过度设计检查”。保留下来的新增模块只有公网运行必需的五类：账号与隔离、资产额度、普通用户全局能力开关、浏览器密钥租约、Windows 启动与备份。能力开关只有一套全局策略，不做用户分组、逐用户权限、角色体系或复杂调度；其余模块也没有抽象成通用 IAM、通用配额平台或通用密钥管理系统。

## 3. 最终结构

```text
网友浏览器
  └─ HTTPS: saber.mashirosaber.work
       └─ Cloudflare Edge / Tunnel
            └─ HTTP: 127.0.0.1:5100
                 └─ Saber Launcher（public profile）
                      ├─ Flask API + Vue 静态文件
                      ├─ Worker
                      ├─ SQLite + 对象目录
                      └─ 仅回环地址可访问的内存密钥服务
```

Cloudflare 负责公网 TLS、域名入口和基础 DDoS 防护。Saber 自己负责账号、会话、CSRF、管理员权限、数据归属和资产额度。两层职责不重复。

## 4. 两种启动模式

| 模式 | 用途 | 认证 | 数据目录 | 端口建议 |
|---|---|---:|---|---:|
| `local` | 原有本地桌面使用 | 否 | 原 `data-v2` | 5000 |
| `public` | 正式公网服务 | 是 | `D:\Saber-Translator\data-public` | 5100 |

这不是复制两套业务代码。两个 profile 只固定本地和公网所需的安全策略，核心前后端仍是同一套代码。上线前需要验证公网规则时，使用临时数据目录启动 `public` 即可，不再增加第三种模式。

## 5. 账号与页面逻辑

### 登录页 `/login`

- 用户名和密码登录
- 成功后建立 7 天 HttpOnly 会话 Cookie
- 写请求同时校验独立 CSRF Token
- 连续失败会按客户端地址和用户名临时限速

### 注册页 `/register`

- 管理员可以设置注册是否必须使用邀请码，默认开启
- 开启时使用管理员创建的一次性邀请码注册，邀请码 7 天有效且只能使用一次
- 关闭时注册页不显示邀请码字段，符合用户名和密码规则即可自由注册
- 注册成功只显示一次 8 个恢复码，可下载为文本文件
- 邀请码不代表用户数量上限；管理员可以持续创建，不存在总名额字段

### 恢复页 `/recover`

- 用户名、一次性恢复码和新密码完成恢复
- 每个恢复码只能使用一次
- 恢复后撤销该用户所有旧会话

### 账户页 `/account`

- 显示当前用户资产使用量与有效额度
- 说明 Provider 密钥只保存在当前浏览器
- 支持修改密码；修改后撤销所有登录会话

### 管理页 `/admin`

- 仅管理员可访问
- 查看账号状态、资产使用量和有效额度
- 修改所有用户统一使用的资产额度
- 控制普通用户能否使用翻译、漫画分析、角色工坊和编辑模式
- 控制普通用户能否调用各检测、OCR 与 LAMA 本地模型
- 控制普通用户能否修改 LAMA“禁用自动缩放”、能否使用并行模式及其深度学习并发上限
- 启用或禁用普通用户
- 创建、查看状态和撤销邀请码
- 开启或关闭“注册必须使用邀请码”
- 为用户生成一次性恢复码
- 页面明确写明“不限制用户数量和书本数量”

### 原有业务页面

- 书架、翻译、阅读器、漫画分析和角色工坊保留原业务流程
- 每个根资源写入创建者 `owner_user_id`
- 列表查询只返回当前用户资源
- URL 中访问他人资源统一返回 404，避免泄露资源是否存在
- 请求体或查询参数中的 `bookId`、`chapterId`、`pageId` 等同样进行归属校验
- Worker 领取任务后携带任务所有者上下文，生成资产和后续写入仍归原用户

### 公网模式关闭的功能

- 插件安装、插件管理和插件执行
- 网页导入
- Ollama、Sakura 等本机 Provider
- 暴露机器名、局域网地址的 server-info

这些功能在 `local` 模式不受影响。

### 普通用户能力控制

- 只在 `public` 模式限制普通用户；管理员和 `local` 模式不受影响
- 页面入口会隐藏或禁用，后端仍独立强制校验，不能通过直接请求绕过
- 关闭某个本地模型后，新建或重试的任务不能调用它；已经在队列中或正在执行的任务不被中途终止
- “并行模式”只有一个开关和一个深度学习并发上限，不拆成检测、OCR、颜色处理、修复四套配置
- 不增加逐用户例外、权限模板、定时策略、自动性能探测或动态降级逻辑

## 6. 资产额度规则

数据库只保存一个额度字段：

- `platform_config.asset_quota_bytes`：所有用户统一使用的额度，初始为 `2147483648`
- `platform_config.registration_requires_invite`：是否要求邀请码，初始为开启
- `platform_config.public_user_policy_json`：普通用户统一使用的功能、模型与性能策略

所有用户使用同一个管理员可配置的额度。写入新资产时，在 SQLite immediate transaction 中读取当前使用量并校验：

```text
当前有效资产字节数 + 本次新资产字节数 <= 用户有效额度
```

超额时返回 HTTP 413，并携带当前使用量、额度和本次大小；已经写到暂存区或对象目录的失败文件会被清理。并发上传通过同一写事务串行核算，避免两个请求同时越过额度。

修改额度不会删除已有资产。若管理员把额度调到低于当前使用量，用户可以读取和删除现有内容，但新的资产写入会被拒绝。

## 7. Provider 密钥

- 密钥在浏览器 IndexedDB 中按用户、业务域和 Provider 保存
- 保存设置时，浏览器把密钥加载到 Launcher 管理的回环内存服务
- API 和 Worker 的任务配置只保存类似 `browser:translation:custom` 的不透明引用
- 数据库、SQLite 备份、任务配置和日志都不保存原始密钥
- 关闭浏览器不会中断已经加载的密钥；Launcher 重启会清空内存，浏览器下次加载设置时会重新加载
- 如果密钥缺失，任务返回可操作的“重新打开设置并保存密钥”错误

## 8. 公网安全边界

- Host 只接受 `saber.mashirosaber.work`、`localhost` 和 `127.0.0.1`
- 会话 Cookie 为 HttpOnly、Secure、SameSite=Lax
- 所有有副作用的已登录请求校验 CSRF
- 设置 CSP、HSTS、`nosniff`、禁止 iframe、禁止不必要浏览器权限和 API `no-store`
- 对外健康检查只返回状态，不返回 epoch、数据路径指纹或机器信息
- 图片设置解压像素阈值，压缩容器限制路径穿越、条目数、展开体积和异常压缩比
- 公网可配置的外部 Provider URL 只允许 HTTPS，并拒绝解析到回环、局域网、链路本地或其他非公网地址
- HTTP 客户端不自动跟随重定向，避免利用公开 Provider 设置绕过目标地址检查

## 9. Windows 部署目录

```text
D:\Saber-Translator\
├─ app\                 # 经过验证的代码和已构建前端
├─ venv\                # 固定 Python 运行环境副本
├─ data-public\         # 正式公网数据，和本地 data-v2 完全分离
├─ backups\
│  ├─ database\         # 最近 7 个通过 integrity_check 的 SQLite 快照
│  └─ objects-current\  # 一份当前资产镜像
└─ FIRST_ADMIN.txt       # 首次管理员凭据，登录并另存恢复码后删除
```

公网数据目录不能指向现有本地 `data-v2`，这样本地使用和网友使用不会混在一起，也可以单独停止或回滚公网实例。

## 10. 启动、健康检查与备份

仓库中的脚本：

- `scripts/public/Start-Public.ps1`：以前台进程方式启动 Launcher，供计划任务托管
- `scripts/public/Stop-Public.ps1`：停止计划任务并精确回收该公网数据目录的 Launcher 进程树
- `scripts/public/Health-Public.ps1`：验证本地 5100 端口和 API 健康响应
- `scripts/public/Backup-Public.ps1`：在线备份 SQLite，并同步一份对象目录镜像
- `scripts/public/Install-ScheduledTasks.ps1`：安装两个当前用户计划任务

计划任务：

- `Saber Translator Public`：当前 Windows 用户登录时启动；Launcher 自己监管 API 和 Worker
- `Saber Translator Public Backup`：每天 04:15 备份

使用登录触发而不是额外安装 Windows 服务包装器，是为了保持当前家用电脑部署简单，并继续使用当前用户已有的模型缓存和 GPU 环境。代价是 Windows 重启后要等该用户登录，公网服务才恢复。

Launcher 对 Windows 切换 TUN 时可能出现的短暂回环探测中断保留约 15 秒容忍窗口；确需恢复 API 时会同时回收虚拟环境包装进程和实际 Python 子进程，避免旧进程残留占用 5100 端口。子进程退出清理若恰逢 SQLite 短暂写锁，会留到下一轮重试，不会因此让整个 Launcher 退出。

备份位于同一块 D 盘，可以处理数据库损坏、误操作和应用升级回滚，但不能处理整块硬盘损坏。这是当前单机、不增加外部存储前提下的明确边界。

本次能力控制只比当前线上数据库多一个全局策略字段。启动时仅允许从线上紧邻版本执行这一次加列升级，以保留已有用户和书本；不引入通用迁移框架，也不兼容更早的废弃数据库结构。

## 11. 开发到上线的完成顺序

1. 增加 profile 与独立端口/数据目录。
2. 增加账号、会话、CSRF、恢复码和管理员入口。
3. 为所有用户根资源增加归属并贯穿 API、后台任务和派生资产。
4. 增加默认 2 GiB 的唯一资产额度和管理员配置。
5. 把公网 Provider 密钥切换为浏览器保存、Launcher 内存租约。
6. 关闭公网高风险功能并增加 SSRF、上传和响应头防护。
7. 增加登录、注册、恢复、账户和管理页面。
8. 增加普通用户全局功能、模型与性能开关，并在前后端双重校验。
9. 更新数据库 revision、OpenAPI 与生成的 TypeScript 合约。
10. 运行后端、前端、迁移、UI 架构和公网隔离测试。
11. 构建前端并发布到 D 盘独立目录。
12. 创建首个管理员，安装启动与备份计划任务。
13. 本地验证登录、管理、额度、重启和备份。
14. 最后由域名持有人在现有 Tunnel 中增加公网 hostname。

## 12. Cloudflare 最后一步

应用部署完成后，只需要在现有 Tunnel 增加一条 Public Hostname：

```text
Hostname: saber.mashirosaber.work
Service type: HTTP
URL: http://127.0.0.1:5100
```

这是面向网友的应用，不应在这个 hostname 前再启用 Cloudflare Access 登录，否则会出现 Cloudflare 登录和 Saber 登录两层账号。保存 hostname 后确认边缘证书可用，并访问：

```text
https://saber.mashirosaber.work/api/v2/system/capabilities
```

应看到 `profile=public`、`requiresAuth=true`，随后访问首页应进入 Saber 登录页。

### TUN 与 Fake-IP 绕过

继续复用现有的 `Cloudflared` Windows 服务，不启动第二套 Docker 连接器。Clash Verge Rev 的**当前活动配置**必须通过增强规则加入：

```yaml
prepend:
  - PROCESS-NAME,cloudflared.exe,DIRECT
  - DOMAIN-SUFFIX,argotunnel.com,DIRECT
  - DOMAIN-SUFFIX,cfargotunnel.com,DIRECT
```

当前活动配置的 `dns.fake-ip-filter` 必须保留原列表并追加：

```yaml
- '*.argotunnel.com'
- '*.cfargotunnel.com'
- 'region1.v2.argotunnel.com'
- 'region2.v2.argotunnel.com'
```

最后两个精确域名不能只依赖第一条通配符：当前 Mihomo 配置中，`region1.v2.argotunnel.com` 和 `region2.v2.argotunnel.com` 是两层子域，实测只写 `*.argotunnel.com` 时仍会得到 `28.0.0.0/8` Fake-IP。修改后重新生成 Clash 配置、重启一次 `Cloudflared`，并确认两个域名解析为真实的 `198.41.x.x` 地址。

## 13. 验收标准

- 5100 端口只绑定 `127.0.0.1`
- 未登录访问业务 API 返回 401
- 非法 Host 返回 400
- 用户 A 看不到、读取不了、修改不了用户 B 的资源
- 无 CSRF 的写请求返回 403
- 初始额度精确为 2 GiB，管理员修改后对所有用户立即生效
- 额度为 1 字节时，创建无资产书本仍成功，上传图片返回 413
- 数据库中不存在浏览器 Provider 原始密钥
- public 页面不显示插件、网页导入和本机 Provider
- 修改密码后旧会话失效
- 管理员关闭邀请码要求后可无邀请码注册，重新开启后再次强制校验邀请码
- 普通用户被关闭的页面入口不可进入，直接请求也返回 403；管理员仍可使用
- 被关闭的检测、OCR 或 LAMA 模型不能通过任务与编辑操作调用
- 锁定 LAMA 自动缩放或关闭并行后，普通用户保存设置也不能绕过管理员策略
- 使用 `Stop-Public.ps1` 停止后，再启动计划任务能够恢复服务；内存密钥需要浏览器重新加载
- SQLite 快照通过 `PRAGMA integrity_check`
- Cloudflare hostname 可从手机网络访问
- TUN 开启、关闭以及关闭后重新开启时，公网健康接口均返回 200，`Cloudflared` 不需要人工重启且不出现 1033

## 14. 仍需知晓但不增加系统的两点

1. 不限制用户数量且每人可用 2 GiB，理论总用量会随真实用户数增长。管理员可以用一次性邀请码控制注册来源，也可以开启自由注册；无论哪种方式都没有用户总数上限，因此应偶尔查看 D 盘剩余空间。
2. 当前备份和正式数据在同一块 D 盘。如果以后确实出现重要用户数据，再把 `backups` 目录同步到另一块硬盘即可；现在不提前建设云备份系统。
