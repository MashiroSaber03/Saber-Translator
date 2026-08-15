# 后端角色、领域与生产文件映射

> 最终状态已按 `current-files.tsv` 同步；审查中删除的冗余文件保留为“删除已验证”。行数按最终工作区重新计算。

本文件由当前 Python AST 导入关系生成。动态配置选择、插件文件扫描和 PyInstaller hidden imports 仍需在逐文件审查时另行核实，不能仅凭本图判定死代码。

## 正式角色入口

| 角色 | 根文件 |
|---|---|
| PASS | `saber_v2.py` |
| PASS | `src/backend_v2/api/entrypoint.py` |
| FIXED | `src/backend_v2/worker/entrypoint.py` |
| FIXED | `src/backend_v2/desktop/entrypoint.py` |
| FIXED | `src/backend_v2/launcher/entrypoint.py` |
| 插件包 | `plugins/*/plugin.py`（运行时按插件契约扫描） |

## 文件归属闭包

| 状态 | 文件 | 行数 | 层/领域 | AST 可达角色 |
|---|---|---:|---|---|
| 删除已验证 | `plugins/pipeline_lifecycle_plugin/__init__.py` | 1 | plugins | 插件包 |
| PASS | `plugins/pipeline_lifecycle_plugin/plugin.py` | 32 | plugins | 插件包 |
| 删除已验证 | `plugins/style_mutation_plugin/__init__.py` | 2 | plugins | 插件包 |
| PASS | `plugins/style_mutation_plugin/plugin.py` | 39 | plugins | 插件包 |
| 删除已验证 | `plugins/text_mutation_plugin/__init__.py` | 2 | plugins | 插件包 |
| PASS | `plugins/text_mutation_plugin/plugin.py` | 35 | plugins | 插件包 |
| PASS | `saber_v2.py` | 22 | entrypoint | 统一命令入口 |
| PASS | `src/backend_v2/__init__.py` | 1 | backend-v2/root | ORPHAN |
| PASS | `src/backend_v2/api/__init__.py` | 1 | backend-v2/api | ORPHAN |
| PASS | `src/backend_v2/api/app.py` | 312 | backend-v2/api | API、统一命令入口 |
| PASS | `src/backend_v2/api/entrypoint.py` | 159 | backend-v2/api | API、统一命令入口 |
| PASS | `src/backend_v2/api/request_helpers.py` | 116 | backend-v2/api | API、统一命令入口 |
| PASS | `src/backend_v2/api/system_routes.py` | 31 | backend-v2/api | API、统一命令入口 |
| PASS | `src/backend_v2/api/web.py` | 47 | backend-v2/api | API、统一命令入口 |
| PASS | `src/backend_v2/checksums.py` | 14 | backend-v2/root | API、统一命令入口、Worker |
| PASS | `src/backend_v2/content/__init__.py` | 1 | backend-v2/content | ORPHAN |
| FIXED | `src/backend_v2/content/image_import.py` | 428 | backend-v2/content | API、统一命令入口、Worker |
| PASS | `src/backend_v2/content/media.py` | 66 | backend-v2/content | API、统一命令入口 |
| FIXED | `src/backend_v2/content/page_style.py` | 202 | backend-v2/content | API、统一命令入口、Worker |
| FIXED | `src/backend_v2/content/repository.py` | 2861 | backend-v2/content | API、统一命令入口、Worker |
| FIXED | `src/backend_v2/content/routes.py` | 735 | backend-v2/content | API、统一命令入口 |
| FIXED | `src/backend_v2/content/translation_constraints.py` | 266 | backend-v2/content | API、桌面 GUI、统一命令入口、Launcher、Worker |
| PASS | `src/backend_v2/desktop/__init__.py` | 1 | backend-v2/desktop | ORPHAN |
| FIXED | `src/backend_v2/desktop/entrypoint.py` | 475 | backend-v2/desktop | 桌面 GUI、统一命令入口 |
| FIXED | `src/backend_v2/desktop/pet.py` | 482 | backend-v2/desktop | 桌面 GUI、统一命令入口 |
| PASS | `src/backend_v2/desktop/pet_state.py` | 148 | backend-v2/desktop | 桌面 GUI、统一命令入口 |
| FIXED | `src/backend_v2/desktop/settings.py` | 183 | backend-v2/desktop | 桌面 GUI、统一命令入口 |
| FIXED | `src/backend_v2/desktop/task_client.py` | 290 | backend-v2/desktop | 桌面 GUI、统一命令入口 |
| PASS | `src/backend_v2/desktop/theme.py` | 186 | backend-v2/desktop | 桌面 GUI、统一命令入口 |
| FIXED | `src/backend_v2/desktop/window.py` | 899 | backend-v2/desktop | 桌面 GUI、统一命令入口 |
| FIXED | `src/backend_v2/dispatch.py` | 59 | backend-v2/root | 统一命令入口 |
| PASS | `src/backend_v2/domain/__init__.py` | 1 | backend-v2/domain | ORPHAN |
| PASS | `src/backend_v2/domain/state_machines.py` | 66 | backend-v2/domain | API、统一命令入口、Worker |
| PASS | `src/backend_v2/import_guard.py` | 37 | backend-v2/root | API、统一命令入口 |
| PASS | `src/backend_v2/insight/__init__.py` | 2 | backend-v2/insight | ORPHAN |
| FIXED | `src/backend_v2/insight/commands.py` | 517 | backend-v2/insight | API、统一命令入口 |
| FIXED | `src/backend_v2/insight/continuation.py` | 4009 | backend-v2/insight | API、统一命令入口、Worker |
| FIXED | `src/backend_v2/insight/derived.py` | 4639 | backend-v2/insight | API、统一命令入口、Worker |
| FIXED | `src/backend_v2/insight/exports.py` | 924 | backend-v2/insight | API、统一命令入口、Worker |
| PASS | `src/backend_v2/insight/gc.py` | 216 | backend-v2/insight | 统一命令入口、Worker |
| PASS | `src/backend_v2/insight/page_schema.py` | 215 | backend-v2/insight | 统一命令入口、Worker |
| FIXED | `src/backend_v2/insight/qa.py` | 1763 | backend-v2/insight | API、统一命令入口、Worker |
| FIXED | `src/backend_v2/insight/repository.py` | 3820 | backend-v2/insight | API、统一命令入口、Worker |
| FIXED | `src/backend_v2/insight/routes.py` | 1212 | backend-v2/insight | API、统一命令入口 |
| PASS | `src/backend_v2/insight/worker.py` | 383 | backend-v2/insight | 统一命令入口、Worker |
| PASS | `src/backend_v2/jobs/__init__.py` | 1 | backend-v2/jobs | ORPHAN |
| PASS | `src/backend_v2/jobs/events.py` | 139 | backend-v2/jobs | API、统一命令入口 |
| FIXED | `src/backend_v2/jobs/repository.py` | 5083 | backend-v2/jobs | API、统一命令入口、Worker |
| FIXED | `src/backend_v2/jobs/retry.py` | 1168 | backend-v2/jobs | API、统一命令入口 |
| FIXED | `src/backend_v2/jobs/routes.py` | 314 | backend-v2/jobs | API、统一命令入口 |
| FIXED | `src/backend_v2/jobs/worker_loop.py` | 1175 | backend-v2/jobs | 统一命令入口、Worker |
| PASS | `src/backend_v2/launcher/__init__.py` | 1 | backend-v2/launcher | ORPHAN |
| FIXED | `src/backend_v2/launcher/entrypoint.py` | 887 | backend-v2/launcher | 桌面 GUI、统一命令入口、Launcher |
| PASS | `src/backend_v2/launcher/windows_job.py` | 117 | backend-v2/launcher | 桌面 GUI、统一命令入口、Launcher |
| PASS | `src/backend_v2/logging_config.py` | 144 | backend-v2/root | API、桌面 GUI、统一命令入口、Launcher、Worker |
| PASS | `src/backend_v2/operations/__init__.py` | 1 | backend-v2/operations | ORPHAN |
| FIXED | `src/backend_v2/operations/executor.py` | 406 | backend-v2/operations | API、统一命令入口、Worker |
| FIXED | `src/backend_v2/operations/repair.py` | 546 | backend-v2/operations | API、统一命令入口、Worker |
| FIXED | `src/backend_v2/operations/repository.py` | 1713 | backend-v2/operations | API、统一命令入口、Worker |
| FIXED | `src/backend_v2/operations/routes.py` | 235 | backend-v2/operations | API、统一命令入口 |
| PASS | `src/backend_v2/paths.py` | 57 | backend-v2/root | API、桌面 GUI、统一命令入口、Launcher、Worker |
| PASS | `src/backend_v2/plugins/__init__.py` | 2 | backend-v2/plugins | ORPHAN |
| FIXED | `src/backend_v2/plugins/agent.py` | 962 | backend-v2/plugins | API、统一命令入口、Worker |
| PASS | `src/backend_v2/plugins/agent_routes.py` | 115 | backend-v2/plugins | API、统一命令入口 |
| FIXED | `src/backend_v2/plugins/agent_tools.py` | 254 | backend-v2/plugins | 统一命令入口、Worker |
| PASS | `src/backend_v2/plugins/agent_worker.py` | 345 | backend-v2/plugins | 统一命令入口、Worker |
| FIXED | `src/backend_v2/plugins/contract.py` | 1115 | backend-v2/plugins | API、统一命令入口、Worker |
| FIXED | `src/backend_v2/plugins/package.py` | 166 | backend-v2/plugins | API、统一命令入口、Worker |
| FIXED | `src/backend_v2/plugins/repository.py` | 844 | backend-v2/plugins | API、统一命令入口、Worker |
| FIXED | `src/backend_v2/plugins/routes.py` | 198 | backend-v2/plugins | API、统一命令入口 |
| FIXED | `src/backend_v2/plugins/runtime.py` | 979 | backend-v2/plugins | 统一命令入口、Worker |
| FIXED | `src/backend_v2/plugins/snapshots.py` | 78 | backend-v2/plugins | API、统一命令入口、Worker |
| PASS | `src/backend_v2/redaction.py` | 185 | backend-v2/root | API、桌面 GUI、统一命令入口、Launcher、Worker |
| PASS | `src/backend_v2/rendering/__init__.py` | 2 | backend-v2/rendering | ORPHAN |
| PASS | `src/backend_v2/rendering/fonts.py` | 135 | backend-v2/rendering | API、统一命令入口、Worker |
| PASS | `src/backend_v2/rendering/service.py` | 170 | backend-v2/rendering | API、统一命令入口、Worker |
| FIXED | `src/backend_v2/runtime_heartbeat.py` | 107 | backend-v2/root | API、统一命令入口、Worker |
| PASS | `src/backend_v2/runtime_identity.py` | 113 | backend-v2/root | API、桌面 GUI、统一命令入口、Launcher、Worker |
| PASS | `src/backend_v2/serialization.py` | 14 | backend-v2/root | API、桌面 GUI、统一命令入口、Launcher、Worker |
| PASS | `src/backend_v2/settings/__init__.py` | 1 | backend-v2/settings | ORPHAN |
| PASS | `src/backend_v2/settings/diagnostics.py` | 689 | backend-v2/settings | API、统一命令入口 |
| FIXED | `src/backend_v2/settings/resolver.py` | 1065 | backend-v2/settings | API、统一命令入口、Worker |
| PASS | `src/backend_v2/settings/routes.py` | 528 | backend-v2/settings | API、统一命令入口 |
| PASS | `src/backend_v2/settings/validation.py` | 953 | backend-v2/settings | API、统一命令入口、Worker |
| PASS | `src/backend_v2/storage/__init__.py` | 1 | backend-v2/storage | ORPHAN |
| PASS | `src/backend_v2/storage/assets.py` | 571 | backend-v2/storage | API、桌面 GUI、统一命令入口、Launcher、Worker |
| PASS | `src/backend_v2/storage/builtin_fonts.py` | 132 | backend-v2/storage | API、桌面 GUI、统一命令入口、Launcher、Worker |
| PASS | `src/backend_v2/storage/consistency.py` | 145 | backend-v2/storage | ORPHAN |
| PASS | `src/backend_v2/storage/database.py` | 128 | backend-v2/storage | API、桌面 GUI、统一命令入口、Launcher、Worker |
| FIXED | `src/backend_v2/storage/defaults.py` | 249 | backend-v2/storage | API、桌面 GUI、统一命令入口、Launcher、Worker |
| FIXED | `src/backend_v2/storage/epochs.py` | 546 | backend-v2/storage | API、桌面 GUI、统一命令入口、Launcher、Worker |
| PASS | `src/backend_v2/storage/lifecycle.py` | 143 | backend-v2/storage | 桌面 GUI、统一命令入口、Launcher |
| PASS | `src/backend_v2/storage/migrations/env.py` | 84 | …889 tokens truncated…sfer/commands.py` | 409 | backend-v2/transfer | API、统一命令入口 |
| FIXED | `src/backend_v2/storage/migrations/versions/v2_foundation_20260810_backend_v2_foundation.py` | 1541 | backend-v2/storage | ORPHAN |
| FIXED | `src/backend_v2/storage/platform_repositories.py` | 1704 | backend-v2/storage | API、统一命令入口、Worker |
| FIXED | `src/backend_v2/storage/schema.py` | 2201 | backend-v2/storage | API、桌面 GUI、统一命令入口、Launcher、Worker |
| PASS | `src/backend_v2/storage/seeding.py` | 186 | backend-v2/storage | API、桌面 GUI、统一命令入口、Launcher、Worker |
| PASS | `src/backend_v2/storage/single_instance.py` | 103 | backend-v2/storage | 桌面 GUI、统一命令入口、Launcher |
| PASS | `src/backend_v2/studio/__init__.py` | 2 | backend-v2/studio | ORPHAN |
| PASS | `src/backend_v2/studio/io.py` | 431 | backend-v2/studio | API、统一命令入口 |
| PASS | `src/backend_v2/studio/media.py` | 90 | backend-v2/studio | API、统一命令入口 |
| PASS | `src/backend_v2/studio/model.py` | 237 | backend-v2/studio | API、统一命令入口 |
| FIXED | `src/backend_v2/studio/pure.py` | 1392 | backend-v2/studio | API、统一命令入口 |
| FIXED | `src/backend_v2/studio/repository.py` | 2775 | backend-v2/studio | API、统一命令入口 |
| FIXED | `src/backend_v2/studio/routes.py` | 847 | backend-v2/studio | API、统一命令入口 |
| FIXED | `src/backend_v2/studio/service.py` | 1274 | backend-v2/studio | API、统一命令入口 |
| PASS | `src/backend_v2/timestamps.py` | 17 | backend-v2/root | API、桌面 GUI、统一命令入口、Launcher、Worker |
| PASS | `src/backend_v2/transfer/__init__.py` | 1 | backend-v2/transfer | ORPHAN |
| FIXED | `src/backend_v2/transfer/commands.py` | 409 | backend-v2/transfer | API、统一命令入口 |
| PASS | `src/backend_v2/transfer/routes.py` | 64 | backend-v2/transfer | API、统一命令入口 |
| PASS | `src/backend_v2/transfer/worker.py` | 614 | backend-v2/transfer | 统一命令入口、Worker |
| PASS | `src/backend_v2/translation/__init__.py` | 2 | backend-v2/translation | ORPHAN |
| PASS | `src/backend_v2/translation/auxiliary.py` | 1463 | backend-v2/translation | API、统一命令入口、Worker |
| PASS | `src/backend_v2/translation/commands.py` | 895 | backend-v2/translation | API、统一命令入口、Worker |
| PASS | `src/backend_v2/translation/interactive_operations.py` | 866 | backend-v2/translation | API、统一命令入口、Worker |
| PASS | `src/backend_v2/translation/pipeline.py` | 3515 | backend-v2/translation | API、统一命令入口、Worker |
| PASS | `src/backend_v2/translation/routes.py` | 220 | backend-v2/translation | API、统一命令入口 |
| PASS | `src/backend_v2/web_import/__init__.py` | 2 | backend-v2/web_import | ORPHAN |
| FIXED | `src/backend_v2/web_import/commands.py` | 1159 | backend-v2/web_import | API、统一命令入口、Worker |
| PASS | `src/backend_v2/web_import/routes.py` | 181 | backend-v2/web_import | API、统一命令入口 |
| FIXED | `src/backend_v2/web_import/worker.py` | 1351 | backend-v2/web_import | 统一命令入口、Worker |
| PASS | `src/backend_v2/worker/__init__.py` | 5 | backend-v2/worker | ORPHAN |
| FIXED | `src/backend_v2/worker/entrypoint.py` | 436 | backend-v2/worker | 统一命令入口、Worker |
| PASS | `src/backend_v2/worker/maintenance.py` | 387 | backend-v2/worker | 统一命令入口、Worker |
| PASS | `src/backend_v2/worker/model_lifecycle.py` | 499 | backend-v2/worker | API、统一命令入口、Worker |
| PASS | `src/core/__init__.py` | 1 | core | ORPHAN |
| FIXED | `src/core/color_extractor.py` | 255 | core | API、统一命令入口、Worker |
| FIXED | `src/core/config_models.py` | 433 | core | API、统一命令入口、Worker |
| FIXED | `src/core/detection.py` | 338 | core | API、统一命令入口、Worker |
| PASS | `src/core/detector/__init__.py` | 33 | core | API、统一命令入口、Worker |
| FIXED | `src/core/detector/aux_yolo.py` | 184 | core | API、统一命令入口、Worker |
| PASS | `src/core/detector/backends/__init__.py` | 1 | core | ORPHAN |
| FIXED | `src/core/detector/backends/ctd_backend.py` | 175 | core | API、统一命令入口、Worker |
| FIXED | `src/core/detector/backends/default_backend.py` | 233 | core | API、统一命令入口、Worker |
| FIXED | `src/core/detector/backends/saber_yolo_backend.py` | 96 | core | API、统一命令入口、Worker |
| FIXED | `src/core/detector/backends/yolo_backend.py` | 221 | core | API、统一命令入口、Worker |
| FIXED | `src/core/detector/base.py` | 209 | core | API、统一命令入口、Worker |
| FIXED | `src/core/detector/data_types.py` | 484 | core | API、统一命令入口、Worker |
| FIXED | `src/core/detector/geometry.py` | 85 | core | API、统一命令入口、Worker |
| FIXED | `src/core/detector/panel_detector.py` | 215 | core | API、统一命令入口、Worker |
| FIXED | `src/core/detector/postprocess.py` | 167 | core | API、统一命令入口、Worker |
| FIXED | `src/core/detector/refinement.py` | 242 | core | API、统一命令入口、Worker |
| FIXED | `src/core/detector/registry.py` | 171 | core | API、统一命令入口、Worker |
| PASS | `src/core/detector/smart_sort.py` | 228 | core | API、统一命令入口、Worker |
| FIXED | `src/core/detector/textline_merge.py` | 285 | core | API、统一命令入口、Worker |
| FIXED | `src/core/inpainting.py` | 353 | core | API、统一命令入口、Worker |
| FIXED | `src/core/large_image_detection.py` | 205 | core | API、统一命令入口、Worker |
| PASS | `src/core/manga_insight/__init__.py` | 1 | core | ORPHAN |
| PASS | `src/core/manga_insight/clients/__init__.py` | 1 | core | ORPHAN |
| 删除已验证 | `src/core/manga_insight/clients/base_client.py` | 113 | core | API、统一命令入口、Worker |
| FIXED | `src/core/manga_insight/clients/image_gen_client.py` | 426 | core | API、统一命令入口、Worker |
| 删除已验证 | `src/core/manga_insight/config/__init__.py` | 12 | core | ORPHAN |
| 删除已验证 | `src/core/manga_insight/config/serialization.py` | 145 | core | API、统一命令入口、Worker |
| FIXED | `src/core/manga_insight/config_models.py` | 400 | core | API、统一命令入口、Worker |
| FIXED | `src/core/manga_insight/embedding_client.py` | 222 | core | API、统一命令入口、Worker |
| 删除已验证 | `src/core/manga_insight/utils/__init__.py` | 21 | core | ORPHAN |
| 删除已验证 | `src/core/manga_insight/utils/json_parser.py` | 91 | core | 统一命令入口、Worker |
| FIXED | `src/core/manga_insight/vlm_client.py` | 211 | core | 统一命令入口、Worker |
| FIXED | `src/core/ocr.py` | 545 | core | API、统一命令入口、Worker |
| FIXED | `src/core/ocr_hybrid_manga_48.py` | 303 | core | API、统一命令入口、Worker |
| FIXED | `src/core/ocr_types.py` | 213 | core | API、统一命令入口、Worker |
| PASS | `src/core/plugin_agent/__init__.py` | 1 | core | ORPHAN |
| PASS | `src/core/plugin_agent/controller.py` | 779 | core | API、统一命令入口、Worker |
| PASS | `src/core/plugin_agent/models.py` | 121 | core | API、统一命令入口、Worker |
| FIXED | `src/core/rendering.py` | 1754 | core | API、统一命令入口、Worker |
| FIXED | `src/core/translation.py` | 708 | core | API、统一命令入口、Worker |
| PASS | `src/core/web_import/__init__.py` | 5 | core | 统一命令入口、Worker |
| PASS | `src/core/web_import/agent.py` | 560 | core | 统一命令入口、Worker |
| PASS | `src/core/web_import/firecrawl_tools.py` | 210 | core | 统一命令入口、Worker |
| 删除已验证 | `src/core/web_import/prompts.py` | 68 | core | 统一命令入口、Worker |
| PASS | `src/interfaces/__init__.py` | 4 | interfaces | ORPHAN |
| FIXED | `src/interfaces/baidu_ocr_interface.py` | 283 | interfaces | API、统一命令入口、Worker |
| FIXED | `src/interfaces/baidu_translate_interface.py` | 102 | interfaces | API、统一命令入口、Worker |
| PASS | `src/interfaces/ctd/__init__.py` | 1 | interfaces | ORPHAN |
| PASS | `src/interfaces/ctd/basemodel.py` | 168 | interfaces | API、统一命令入口、Worker |
| PASS | `src/interfaces/ctd/detector.py` | 39 | interfaces | API、统一命令入口、Worker |
| PASS | `src/interfaces/ctd/utils/__init__.py` | 1 | interfaces | ORPHAN |
| PASS | `src/interfaces/ctd/utils/db_utils.py` | 118 | interfaces | API、统一命令入口、Worker |
| PASS | `src/interfaces/ctd/utils/imgproc_utils.py` | 31 | interfaces | API、统一命令入口、Worker |
| PASS | `src/interfaces/ctd/utils/yolov5_utils.py` | 48 | interfaces | API、统一命令入口、Worker |
| PASS | `src/interfaces/ctd/yolov5/__init__.py` | 1 | interfaces | ORPHAN |
| PASS | `src/interfaces/ctd/yolov5/common.py` | 148 | interfaces | API、统一命令入口、Worker |
| PASS | `src/interfaces/ctd/yolov5/yolo.py` | 272 | interfaces | API、统一命令入口、Worker |
| PASS | `src/interfaces/default/__init__.py` | 1 | interfaces | API、统一命令入口、Worker |
| PASS | `src/interfaces/default/DBHead.py` | 41 | interfaces | API、统一命令入口、Worker |
| PASS | `src/interfaces/default/DBNet_resnet34.py` | 110 | interfaces | API、统一命令入口、Worker |
| PASS | `src/interfaces/default/imgproc.py` | 29 | interfaces | API、统一命令入口、Worker |
| FIXED | `src/interfaces/lama_interface.py` | 393 | interfaces | API、统一命令入口、Worker |
| FIXED | `src/interfaces/lama_mpe_interface.py` | 823 | interfaces | API、统一命令入口、Worker |
| FIXED | `src/interfaces/manga_ocr_interface.py` | 147 | interfaces | API、统一命令入口、Worker |
| PASS | `src/interfaces/ocr_48px/__init__.py` | 24 | interfaces | API、统一命令入口、Worker |
| PASS | `src/interfaces/ocr_48px/core.py` | 372 | interfaces | API、统一命令入口、Worker |
| FIXED | `src/interfaces/ocr_48px/interface.py` | 800 | interfaces | API、统一命令入口、Worker |
| PASS | `src/interfaces/ocr_48px/xpos.py` | 65 | interfaces | API、统一命令入口、Worker |
| FIXED | `src/interfaces/paddle_ocr_onnx_interface.py` | 358 | interfaces | API、统一命令入口、Worker |
| FIXED | `src/interfaces/paddleocr_vl_interface.py` | 387 | interfaces | API、统一命令入口、Worker |
| FIXED | `src/interfaces/vision_interface.py` | 151 | interfaces | API、统一命令入口、Worker |
| FIXED | `src/interfaces/youdao_translate_interface.py` | 102 | interfaces | API、统一命令入口、Worker |
| PASS | `src/shared/__init__.py` | 1 | shared | API、桌面 GUI、统一命令入口、Launcher、Worker |
| FIXED | `src/shared/ai_adapters.py` | 73 | shared | API、统一命令入口、Worker |
| FIXED | `src/shared/ai_providers.py` | 341 | shared | API、统一命令入口、Worker |
| FIXED | `src/shared/ai_transport.py` | 1203 | shared | API、统一命令入口、Worker |
| FIXED | `src/shared/constants.py` | 186 | shared | API、桌面 GUI、统一命令入口、Launcher、Worker |
| FIXED | `src/shared/http_config.py` | 53 | shared | API、统一命令入口、Worker |
| FIXED | `src/shared/image_helpers.py` | 43 | shared | API、统一命令入口、Worker |
| PASS | `src/shared/memory_errors.py` | 57 | shared | API、统一命令入口、Worker |
| FIXED | `src/shared/openai_execution.py` | 448 | shared | API、统一命令入口、Worker |
| FIXED | `src/shared/openai_helpers.py` | 87 | shared | API、统一命令入口、Worker |
| FIXED | `src/shared/openai_options.py` | 267 | shared | API、统一命令入口、Worker |
| FIXED | `src/shared/openai_rate_limits.py` | 157 | shared | API、统一命令入口、Worker |
| PASS | `src/shared/path_helpers.py` | 69 | shared | API、桌面 GUI、统一命令入口、Launcher、Worker |
| PASS | `src/shared/text_style_defaults.py` | 103 | shared | API、桌面 GUI、统一命令入口、Launcher、Worker |
| PASS | `src/utils/__init__.py` | 1 | utils | ORPHAN |
| PASS | `src/utils/image_rearrange.py` | 410 | utils | API、统一命令入口、Worker |

## 初始 AST 孤立项复核

- [x] 各层 `__init__.py`：逐一核对后仅保留无副作用的常规 Python 包边界/职责说明，不导出旧应用对象、不触发模型或运行时初始化；这类文件不是应按静态 import 计数删除的业务实现。
- [x] `src/backend_v2/storage/consistency.py`：由 `scripts/check_v2_consistency.py` 正式命令入口和 Stage1 平台测试直接使用。
- [x] `src/backend_v2/storage/migrations/env.py` 与 `versions/v2_foundation_20260810_backend_v2_foundation.py`：是 Alembic 动态装载的当前唯一 foundation 入口，revision 由平台与存储契约测试校验。
- [x] `src/core/manga_insight/config/__init__.py` 与 `src/core/manga_insight/utils/__init__.py`：当前工作区已删除，删除原因与替代关系记录在 `main-removed-files.tsv`，不再属于现存文件。

## 使用规则

- 角色可达只证明存在静态导入链，不证明职责正确或运行路径可达。
- 每个领域文件仍须完整阅读，并从路由/命令追到仓储、事务、Worker handler 与终态。
- `src/core/interfaces/shared/utils` 必须额外证明没有旧应用编排和前端状态职责。

