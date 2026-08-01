import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import ProductSelectableImageGrid from '@/components/product/ProductSelectableImageGrid.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
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
        engineDisplayName: 'AI Agent',
        error: null,
        extractResult: createExtractResult(),
        isAllSelected: false,
        selectedCount: 0,
        selectedPages: new Set<number>(),
        status: 'extracted',
      },
    })

    expect(wrapper.findComponent(ProductSelectableImageGrid).exists()).toBe(true)

    const grid = wrapper.getComponent(ProductSelectableImageGrid)
    expect(grid.props('ariaLabel')).toBe('网页导入图片选择')
    expect(grid.props('items')[0]).toMatchObject({
      id: 1,
      label: '第 1 页',
      selected: false,
    })

    const firstCheckbox = wrapper
      .findAllComponents(UiCheckbox)
      .find(checkbox => checkbox.props('ariaLabel') === '选择第 1 页')
    expect(firstCheckbox).toBeDefined()
    grid.vm.$emit('toggle', 1)

    expect(wrapper.emitted('togglePage')?.[0]).toEqual([1])
    expect(wrapper.find('.image-item').exists()).toBe(false)
  })

  it('renders result selection controls through the product action row', () => {
    const wrapper = mount(WebImportResultsGrid, {
      props: {
        downloadProgress: { current: 0, total: 0 },
        engineDisplayName: 'AI Agent',
        error: null,
        extractResult: createExtractResult(),
        isAllSelected: true,
        selectedCount: 2,
        selectedPages: new Set<number>([1, 2]),
        status: 'extracted',
      },
    })

    const actionRow = wrapper.getComponent(ProductActionRow)

    expect(actionRow.props('ariaLabel')).toBe('网页导入结果选择')
    expect(actionRow.props('justify')).toBe('start')
    expect(wrapper.find('.select-control').exists()).toBe(false)
    expect(wrapper.text()).toContain('已选: 2 张')
  })

  it('renders result count and engine metadata through product chips', () => {
    const wrapper = mount(WebImportResultsGrid, {
      props: {
        downloadProgress: { current: 0, total: 0 },
        engineDisplayName: 'AI Agent',
        error: null,
        extractResult: createExtractResult(),
        isAllSelected: true,
        selectedCount: 2,
        selectedPages: new Set<number>([1, 2]),
        status: 'extracted',
      },
    })

    const metaChips = wrapper.getComponent(ProductChipList)
    expect(metaChips.props('ariaLabel')).toBe('网页导入结果元信息')
    expect(metaChips.props('items')).toEqual([
      {
        id: 'page-count',
        label: '共 2 张',
        tone: 'neutral',
      },
      {
        id: 'engine',
        label: '引擎: AI Agent',
        tone: 'neutral',
      },
    ])
    expect(wrapper.find('.result-count').exists()).toBe(false)
    expect(wrapper.find('.result-engine').exists()).toBe(false)
  })

  it('renders result titles through the shared product section header', () => {
    const wrapper = mount(WebImportResultsGrid, {
      props: {
        downloadProgress: { current: 0, total: 0 },
        engineDisplayName: 'AI Agent',
        error: null,
        extractResult: createExtractResult(),
        isAllSelected: true,
        selectedCount: 2,
        selectedPages: new Set<number>([1, 2]),
        status: 'extracted',
      },
    })
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/web-import/WebImportResultsGrid.vue'),
      'utf8',
    )

    const header = wrapper.getComponent(ProductSectionHeader)
    expect(header.props()).toMatchObject({
      iconName: 'book-open',
      size: 'sm',
      title: '《测试漫画》- 第一话',
    })
    expect(header.findComponent(ProductChipList).exists()).toBe(true)
    expect(source).toContain("import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'")
    expect(source).not.toContain('class="result-header"')
    expect(source).not.toContain('class="result-title"')
  })

  it('keeps result structure hooks owned by the WebImport results grid', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/web-import/WebImportResultsGrid.vue'),
      'utf8',
    )

    expect(source).toContain('web-import-results-grid__section')
    expect(source).toContain('web-import-results-grid__selected-count')
    expect(source).toContain('web-import-results-grid__progress-section')
    expect(source).not.toContain('class="result-section"')
    expect(source).not.toContain('class="selected-count"')
    expect(source).not.toContain('class="progress-section"')
  })

  it('inherits responsive result-header behavior from the shared section header', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/web-import/WebImportResultsGrid.vue'),
      'utf8',
    )
    const sectionHeaderSource = readFileSync(
      resolve(process.cwd(), 'src/components/product/ProductSectionHeader.vue'),
      'utf8',
    )

    expect(source).toContain('ProductSectionHeader')
    expect(source).not.toContain('result-header')
    expect(source).not.toContain('result-title')
    expect(source).not.toContain('result-meta')
    expect(source).not.toContain('result-count')
    expect(source).not.toContain('result-engine')
    expect(sectionHeaderSource).toMatch(/\.product-section-header__title\s*\{[\s\S]*overflow-wrap:\s*anywhere/)
  })

  it('renders extraction errors through the product status banner', () => {
    const wrapper = mount(WebImportResultsGrid, {
      props: {
        downloadProgress: { current: 0, total: 0 },
        engineDisplayName: '',
        error: '提取失败',
        extractResult: null,
        isAllSelected: false,
        selectedCount: 0,
        selectedPages: new Set<number>(),
        status: 'error',
      },
    })

    const banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props('tone')).toBe('danger')
    expect(wrapper.text()).toContain('提取失败')
    expect(wrapper.find('.error-section').exists()).toBe(false)
  })

  it('exposes download progress through progressbar semantics', () => {
    const wrapper = mount(WebImportResultsGrid, {
      props: {
        downloadProgress: { current: 2, total: 4 },
        engineDisplayName: '',
        error: null,
        extractResult: null,
        isAllSelected: false,
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
    expect(wrapper.findComponent(UiProgressBar).exists()).toBe(true)
    expect(wrapper.find('.progress-bar').exists()).toBe(false)
    expect(wrapper.find('.progress-fill').exists()).toBe(false)
  })
})
