import type { V2Job } from '@/api/v2/jobs'

const JOB_KIND_LABELS = {
  translation: '翻译',
  remove_text: '去除文字',
  detect: '文本检测',
  style_apply: '应用样式',
  text_import: '导入文本',
  container_import: '导入文件',
  web_extract: '网页提取',
  web_import_commit: '网页入库',
  export: '导出',
  insight_analysis: '漫画分析',
  insight_export: '分析报告导出',
  vector_rebuild: '重建向量索引',
  continuation: '漫画续写',
  derived_rebuild: '重建分析内容',
  plugin_agent: '插件助手',
} satisfies Readonly<Record<V2Job['kind'], string>>

const STEP_KIND_LABELS: Readonly<Record<string, string>> = {
  detect: '文本检测',
  ocr: '文字识别',
  color: '颜色分析',
  auto_terms: '术语提取',
  translate: '文本翻译',
  hq_translate: '高质量翻译',
  proofread: 'AI 校对',
  repair: '文字修复',
  render: '排版渲染',
  save: '保存结果',
  publish_clean: '发布去字图片',
  style_apply_document: '应用页面样式',
  text_import_apply: '写入导入文本',
  container_scan: '扫描导入文件',
  container_import_page: '导入页面',
  export_package: '打包导出',
  web_extract_scan: '扫描网页内容',
  web_extract_page: '提取网页图片',
  web_extract_finalize: '完成网页提取',
  web_extract_auto_commit: '自动写入章节',
  web_import_commit_page: '导入网页页面',
  web_import_commit_finalize: '完成网页入库',
  insight_analyze_page: '分析漫画页面',
  insight_validate_run: '校验分析结果',
  insight_stage_compressed_context: '生成压缩上下文',
  insight_stage_overview_no_spoiler: '生成无剧透概览',
  insight_stage_overview_story_summary: '生成剧情总结',
  insight_stage_timeline: '生成时间线',
  insight_stage_vectors: '构建向量索引',
  insight_publish_run: '发布分析结果',
  insight_build_overview: '重建作品概览',
  insight_build_compressed_context: '重建压缩上下文',
  insight_build_timeline: '重建时间线',
  insight_build_vectors: '重建向量索引',
  insight_export_report: '生成分析报告',
  continuation_generate_script: '生成续写脚本',
  continuation_generate_page: '生成续写分镜',
  continuation_generate_image: '生成续写图片',
  continuation_generate_character_sheet: '生成角色三视图',
  continuation_export: '导出续写内容',
  plugin_agent_execute: '执行插件助手',
}

const EVENT_TYPE_LABELS = {
  job_created: '创建任务',
  job_reordered: '调整队列顺序',
  job_started: '开始执行',
  job_request_pause: '请求暂停',
  job_request_cancel: '请求取消',
  job_resume: '恢复任务',
  job_continue: '从检查点继续',
  job_paused: '任务已暂停',
  job_cancelled: '任务已取消',
  job_interrupted: '任务已中断',
  job_finished: '任务已完成',
  job_failed: '任务失败',
  chapter_write_lock_acquired: '获得章节写入锁',
  step_started: '步骤开始',
  step_checkpointed: '保存步骤检查点',
  step_completed: '步骤完成',
  pipeline_progress: '流水线进度更新',
  page_completed: '页面完成',
  page_failed: '页面失败',
  page_skipped: '跳过页面',
  plugin_stage_completed: '插件阶段完成',
  plugin_log: '插件日志',
  plugin_hook_failed: '插件钩子失败',
  plugin_hook_completed: '插件钩子完成',
  plugin_agent_state: '插件助手状态更新',
  plugin_agent_assistant_delta: '插件助手回复中',
  plugin_agent_assistant: '插件助手回复完成',
  plugin_agent_tool_call: '插件助手调用工具',
  plugin_agent_tool_result: '插件工具返回结果',
  plugin_agent_validation: '插件校验结果',
  plugin_agent_done: '插件助手完成',
  plugin_agent_error: '插件助手错误',
  web_import_agent_log: '网页导入助手日志',
} as const

export const TASK_EVENT_TYPES = Object.freeze(Object.keys(EVENT_TYPE_LABELS))

function unknownLabel(category: string, value: string): string {
  const normalized = value.trim()
  return normalized ? `未知${category}（${normalized}）` : `未知${category}`
}

export function jobKindLabel(kind: string): string {
  return (JOB_KIND_LABELS as Readonly<Record<string, string>>)[kind] ?? unknownLabel('任务', kind)
}

export function stepKindLabel(kind: string): string {
  const layer = /^insight_build_layer_(\d+)$/.exec(kind)
  if (layer) return `构建分析层 ${layer[1]}`
  return STEP_KIND_LABELS[kind] ?? unknownLabel('步骤', kind)
}

export function eventTypeLabel(type: string): string {
  return (EVENT_TYPE_LABELS as Readonly<Record<string, string>>)[type] ?? unknownLabel('事件', type)
}

export function formatTaskDuration(durationMs: number | null): string {
  if (durationMs === null) return '—'
  const milliseconds = durationMs
  if (milliseconds < 1000) return `${milliseconds} 毫秒`
  if (milliseconds < 60_000) {
    return `${Number((milliseconds / 1000).toFixed(1))} 秒`
  }

  const totalSeconds = Math.round(milliseconds / 1000)
  const seconds = totalSeconds % 60
  const totalMinutes = Math.floor(totalSeconds / 60)
  if (totalMinutes < 60) return `${totalMinutes} 分 ${seconds} 秒`

  const minutes = totalMinutes % 60
  const hours = Math.floor(totalMinutes / 60)
  return `${hours} 小时 ${minutes} 分 ${seconds} 秒`
}
