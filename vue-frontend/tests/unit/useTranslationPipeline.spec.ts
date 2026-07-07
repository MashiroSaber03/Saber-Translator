import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'

const {
  executePipelineMock,
  cancelPipelineMock,
} = vi.hoisted(() => ({
  executePipelineMock: vi.fn(),
  cancelPipelineMock: vi.fn(),
}))

vi.mock('@/composables/translation/core/pipeline', () => ({
  usePipeline: () => ({
    progress: { value: {} },
    isExecuting: { value: false },
    isTranslating: { value: false },
    progressPercent: { value: 0 },
    execute: executePipelineMock,
    cancel: cancelPipelineMock,
  }),
}))

describe('useTranslationPipeline', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    executePipelineMock.mockReset()
    cancelPipelineMock.mockReset()
  })

  it('restores the current image index when single-page translation throws', async () => {
    const imageStore = useImageStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,one')
    imageStore.addImage('page-2.png', 'data:image/png;base64,two')
    imageStore.setCurrentImageIndex(0)

    executePipelineMock.mockRejectedValueOnce(new Error('pipeline failed'))

    const { useTranslation } = await import('@/composables/useTranslationPipeline')
    const translation = useTranslation()

    await expect(
      translation.translatePages([1], 'standard')
    ).rejects.toThrow('pipeline failed')

    expect(imageStore.currentImageIndex).toBe(0)
  })

  it('keeps translation pipeline source comments focused on current workflow contracts', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/composables/useTranslationPipeline.ts'), 'utf8')

    for (const staleNarration of [
      '翻译功能组合式函数',
      '导入管线和模式配置',
      '重新导出类型供外部使用',
      '// ============================================================',
      '辅助函数',
      '组合式函数',
      '便捷方法',
      '重新翻译失败图片',
      '使用已有气泡框翻译',
      '返回',
      '@param',
      '@returns',
      '@example',
    ]) {
      expect(source).not.toContain(staleNarration)
    }

    expect(source).toContain('pipeline.execute(config)')
    expect(source).toContain('imageStore.setCurrentImageIndex(originalIndex)')
  })
})
