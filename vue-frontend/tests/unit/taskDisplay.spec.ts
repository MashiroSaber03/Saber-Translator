import { describe, expect, it } from 'vitest'
import {
  eventTypeLabel,
  formatByteSize,
  formatTaskDuration,
  jobKindLabel,
  resourceRoleLabel,
  stepKindLabel,
} from '@/utils/taskDisplay'

describe('task display labels', () => {
  it('localizes every category without changing its contract value', () => {
    expect(jobKindLabel('translation')).toBe('翻译')
    expect(jobKindLabel('insight_analysis')).toBe('漫画分析')
    expect(stepKindLabel('translate')).toBe('文本翻译')
    expect(stepKindLabel('insight_analyze_page')).toBe('分析漫画页面')
    expect(stepKindLabel('insight_build_layer_3')).toBe('构建分析层 3')
    expect(eventTypeLabel('job_finished')).toBe('任务已完成')
    expect(resourceRoleLabel('text_mask')).toBe('文字遮罩')
  })

  it('keeps an explicit diagnostic fallback for future backend identifiers', () => {
    expect(jobKindLabel('new_job')).toBe('未知任务（new_job）')
    expect(stepKindLabel('new_step')).toBe('未知步骤（new_step）')
    expect(eventTypeLabel('new_event')).toBe('未知事件（new_event）')
    expect(resourceRoleLabel('new_resource')).toBe('未知资源（new_resource）')
  })

  it('formats durations and resource sizes for people', () => {
    expect(formatTaskDuration(null)).toBe('—')
    expect(formatTaskDuration(842)).toBe('842 毫秒')
    expect(formatTaskDuration(20_340)).toBe('20.3 秒')
    expect(formatTaskDuration(136_400)).toBe('2 分 16 秒')
    expect(formatTaskDuration(3_726_000)).toBe('1 小时 2 分 6 秒')
    expect(formatByteSize(842)).toBe('842 B')
    expect(formatByteSize(2_048)).toBe('2 KB')
    expect(formatByteSize(1_572_864)).toBe('1.5 MB')
  })
})
