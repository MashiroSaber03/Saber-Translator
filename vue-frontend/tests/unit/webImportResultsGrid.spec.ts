import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import WebImportResultsGrid from '@/components/translate/web-import/WebImportResultsGrid.vue'
import type { ExtractResult } from '@/types/webImport'

function createExtractResult(): ExtractResult {
  return {
    success: true,
    comicTitle: '测试漫画',
    chapterTitle: '第一话',
    totalPages: 2,
    sourceUrl: 'https://example.com/chapter',
    engine: 'ai-agent',
    pages: [
      { pageNumber: 1, imageUrl: 'https://example.com/1.jpg' },
      { pageNumber: 2, imageUrl: 'https://example.com/2.jpg' },
    ],
  }
}

describe('WebImportResultsGrid', () => {
  it('uses checkbox semantics for whole-card page selection', async () => {
    const wrapper = mount(WebImportResultsGrid, {
      props: {
        downloadProgress: { current: 0, total: 0 },
        downloadProgressPercent: 0,
        engineDisplayName: 'AI Agent',
        error: null,
        extractResult: createExtractResult(),
        isAllSelected: false,
        previewUrlFor: (url: string) => url,
        selectedCount: 0,
        selectedPages: new Set<number>(),
        status: 'extracted',
      },
    })

    const firstCard = wrapper.get('.image-item')
    expect(firstCard.element.tagName).toBe('LABEL')

    const firstCheckbox = wrapper.get('input[aria-label="选择第 1 页"]')
    await firstCheckbox.setValue(true)

    expect(wrapper.emitted('togglePage')?.[0]).toEqual([1])
  })

  it('exposes download progress through progressbar semantics', () => {
    const wrapper = mount(WebImportResultsGrid, {
      props: {
        downloadProgress: { current: 2, total: 4 },
        downloadProgressPercent: 50,
        engineDisplayName: '',
        error: null,
        extractResult: null,
        isAllSelected: false,
        previewUrlFor: (url: string) => url,
        selectedCount: 0,
        selectedPages: new Set<number>(),
        status: 'downloading',
      },
    })

    const progressbar = wrapper.get('[role="progressbar"]')
    expect(progressbar.attributes('aria-valuemin')).toBe('0')
    expect(progressbar.attributes('aria-valuemax')).toBe('4')
    expect(progressbar.attributes('aria-valuenow')).toBe('2')
    expect(progressbar.attributes('aria-label')).toBe('网页导入下载进度')
  })
})
