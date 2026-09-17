# 存储版本与升级维护

本机制从 **3.5.0** 开始。`version.json` 是程序和存储版本的唯一发布来源；严格比较三段数字，不把 `3.5.1` 归并到 `3.5.0`。数据库在 `schema_metadata.storage_version` 中记录实际版本，运行模式仍由 `runtime_profile` 固定。

无版本旧库、数据库缺失但仍有业务文件的目录、较新版本、缺少完整迁移链及模式不匹配均拒绝启动。程序不会给旧库补版本号，也不会删除旧库；请保留旧目录并选择新的数据目录。

## 使用方式

正常启动 Desktop 或 Launcher 自动完成存储准备。Desktop 在读取桌面配置前显示准备进度，并持有整个桌面生命周期的数据目录锁。API/Worker 只校验当前格式，不执行迁移。

源码运行：

```powershell
python saber_v2.py --version
python saber_v2.py --role storage-migrator --action check --data-dir D:\SaberData
python saber_v2.py --role storage-migrator --action upgrade --data-dir D:\SaberData
```

发行包将 `python saber_v2.py` 替换成 `Saber-Translator.exe`。public 模式必须显式提供 `--profile public --data-dir ...`。

`check` 只读，不创建数据目录、锁或备份。`upgrade` 独占运行，完成实际转换后返回 `awaiting_health`，保留备份等待正常 Launcher 首次确认 API 和 Worker 就绪。独立 CLI 不假装已经完成运行验收。

## 实际转换及自动清理

对于 `D:\SaberData`，控制目录是同级的 `D:\.SaberData.saber-storage`。其中包含独占锁、按 PID 与创建时间核验的进程记录、初始化临时目录及 `operations/<操作 UUID>/state.json`。Launcher/桌面日志也存放在控制目录的 `logs` 中，避免 Windows 日志句柄阻止数据根重命名。API/Worker 日志仍在数据根的 `logs` 中。

实际迁移先创建独立完整副本，SQLite 使用 backup API 获取包含 WAL 的一致性内容；文件逐个校验。运行目录不复制。副本中的旧未完成任务结束为 `STORAGE_UPGRADE_INTERRUPTED`，完成成果与历史保留，不自动重新提交模型调用。每步转换与校验成功后才更新副本版本。

目录切换使用持久阶段记录和目录身份标记。原数据移入本次 `backup`，候选副本移入数据根。API 只开放健康检查，Worker 暂不执行任务或维护；两个服务就绪后写入 `committed`，开放业务并立即自动清理。

清理删除本次及历次已提交操作的全部受管理备份、工作副本和恢复废弃副本，保留小型状态记录。没有保留份数设置，也没有确认弹窗。文件占用等清理失败记录在 `cleanup_error` 中，下次正常启动后重试；提交成功后不因清理失败回滚。用户自己创建的备份不属于清理范围。

提交前转换或启动失败会恢复旧数据并结束本次启动。复制/切换中断后先恢复，提示重新启动，不无限重试。已经安装但尚未启动过后端的候选可以在下一次启动继续验收；启动验收中断则回滚。不要手工删除未决操作的状态或身份文件。

## 每次发版需要修改什么

当前 `MIGRATIONS` 为空：3.5.0 是首个基线，不提供无版本数据的生产迁移。测试中的 3.5.1/3.5.2 是隔离样例，不是已发布的兼容承诺。

1. 更新根 `version.json`，发行标签必须完全对应，例如 `v3.5.1`。CI 同时检查版本清单、标签和对应 SQL 契约文件，PyInstaller 打包版本清单与固定契约。
2. 在 `src/storage_migrator/schemas/` 新增目标版本的完整 SQLite DDL。历史文件发布后冻结，不能随着当前 SQLAlchemy metadata 重新生成。
3. 在独立迁移目录新增固定的转换函数及校验函数，并向 `registry.MIGRATIONS` 追加相邻版本步骤。函数只操作传入的副本目录，不引用当前 Repository、模型或插件执行器，不访问网络模型服务。
4. 结构和所有持久化内容都未变化时，显式声明 `Migration("3.5.0", "3.5.1")`；目标 DDL 应保持同样的结构。纯空迁移链校验通过后，在一个事务中仅推进最终版本，不做完整备份、不主动终止任务。
5. 有实际变更时，声明 `Migration("3.5.0", "3.5.1", convert, validate)`。转换函数覆盖本次受到影响的 DDL、JSON、文件路径、插件格式、任务历史、渲染引用与向量代次。通用结构/JSON/资产校验不能代替该步的业务保留性断言。旧向量不兼容时标记需重建，不能把缺失索引标为 ready。
6. 为以后可能作为升级起点的新版本登记固定的 `SOURCE_PREPARERS` 规则，收敛该版本的未完成 Job/Operation/进度/执行身份。3.5.0 的规则保存在 `v3_5_0.py`。若规则未变化，可显式复用原固定函数；若已变化，新增版本规则，不改写历史规则。混合链只在完整副本的起点终止旧工作，之后所有步骤必须保持非运行状态。
7. 增加该版本的数据样例及保留性测试，覆盖直接升级、跨版本链、损坏输入、启动失败、目录切换中断和清理失败重试。修改渲染文档时同步处理版本指针并建立所需的新请求；旧任务检查点不能作为跨实际转换继续执行的依据。

同时在 `VERSION_VALIDATORS` 登记该版本的固定内容校验，供版本相同的启动校验和每步转换完成后的校验共同使用。3.5.0 已包括桌面配置、任务进度，以及实际转换时的插件包哈希和 ready 向量集合检查。发行测试会检查所有已声明版本到当前版本的完整路径、DDL、源版本收敛规则和内容校验是否齐全。

提交后的运行准入以已落盘的 `committed` 状态为准；即使 Windows 文件占用暂时阻止删除门禁或目录标记，也不会让已成功提交的升级一直停在不可用状态。这些标记与备份一起记录待清理并重试。候选安装后的再次校验、桌面配置加载等提交前失败也纳入回滚；恢复记录或门禁记录缺失时明确拒绝自动初始化。

运行核心验证：

```powershell
python -m pytest tests_backend/v2/test_storage_migrator.py tests_backend/v2/test_stage1_platform.py tests_backend/v2/test_role_boundaries.py -q
python -m pytest tests_backend/v2 -q
python .github/scripts/probe_package.py dist/Saber-Translator/Saber-Translator.exe
```

首版应在真实发行包上再次验证入口与资源完整性。源码回归不能替代 CPU/GPU 发行包验收。
