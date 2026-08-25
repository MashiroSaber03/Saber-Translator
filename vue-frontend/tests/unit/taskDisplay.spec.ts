import { describe, expect, it } from 'vitest'
import {
  TASK_EVENT_TYPES,
  eventTypeLabel,
  formatTaskDuration,
  jobKindLabel,
  stepKindLabel,
} from '@/utils/taskDisplay'

describe('task display labels', () => {
  it('localizes every category without changing its contract value', () => {
    expect(jobKindLabel('translation')).toBe('翻译')
    expect(jobKindLabel('insight_analysis')).toBe('漫画分析')
    expect(stepKindLabel('translate')).toBe('文本翻译')
    expect(stepKindLabel('insight_analyze_page')).toBe('分析漫画页面')
    expect(stepKindLabel('insight_build_layer_3')).toBe('构建分析层 4')
    expect(eventTypeLabel('job_finished')).toBe('任务已完成')
    expect(TASK_EVENT_TYPES).toContain('plugin_agent_tool_result')
    expect(TASK_EVENT_TYPES).toContain('web_import_agent_log')
  })

  it('keeps an explicit diagnostic fallback for future backend identifiers', () => {
    expect(jobKindLabel('new_job')).toBe('未知任务（new_job）')
    expect(stepKindLabel('new_step')).toBe('未知步骤（new_step）')
    expect(eventTypeLabel('new_event')).toBe('未知事件（new_event）')
  })

  it('formats durations for people', () => {
    expect(formatTaskDuration(null)).toBe('—')
    expect(formatTaskDuration(842)).toBe('842 毫秒')
    expect(formatTaskDuration(20_340)).toBe('20.3 秒')
    expect(formatTaskDuration(136_400)).toBe('2 分 16 秒')
    expect(formatTaskDuration(3_726_000)).toBe('1 小时 2 分 6 秒')
  })
})
