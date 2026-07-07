import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import ImageResultDisplay from '@/components/translate/ImageResultDisplay.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductScrollStack from '@/components/product/ProductScrollStack.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiField from '@/components/ui/UiField.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { useImageStore } from '@/stores/imageStore'

const exportImportMock = vi.hoisted(() => ({
  state: {
    isDownloading: true,
    downloadProgressText: '正在打包下载',
    downloadProgress: 42,
  },
  downloadCurrentImage: vi.fn(),
  downloadAllImages: vi.fn(),
  exportText: vi.fn(),
  importText: vi.fn(),
}))

vi.mock('@/composables/useExportImport', () => ({
  useExportImport: () => ({
    isDownloading: { value: exportImportMock.state.isDownloading },
    downloadProgressText: { value: exportImportMock.state.downloadProgressText },
    downloadProgress: { value: exportImportMock.state.downloadProgress },
    downloadCurrentImage: exportImportMock.downloadCurrentImage,
    downloadAllImages: exportImportMock.downloadAllImages,
    exportText: exportImportMock.exportText,
    importText: exportImportMock.importText,
  }),
}))

describe('ImageResultDisplay', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    exportImportMock.state.isDownloading = true
    exportImportMock.state.downloadProgressText = '正在打包下载'
    exportImportMock.state.downloadProgress = 42
    exportImportMock.downloadCurrentImage.mockReset()
    exportImportMock.downloadAllImages.mockReset()
    exportImportMock.exportText.mockReset()
    exportImportMock.importText.mockReset()
  })

  it('renders download progress through the shared progress primitive', () => {
    const imageStore = useImageStore()
    imageStore.addImage('page.png', 'data:image/png;base64,aW1hZ2U=')

    const wrapper = mount(ImageResultDisplay)

    const progress = wrapper.getComponent(UiProgressBar)
    expect(progress.props('value')).toBe(42)
    expect(progress.text()).toContain('正在打包下载')
    expect(wrapper.find('.progress').exists()).toBe(false)
  })

  it('uses the native product select for fixed download formats', async () => {
    exportImportMock.state.isDownloading = false
    const imageStore = useImageStore()
    imageStore.addImage('page.png', 'data:image/png;base64,aW1hZ2U=')

    const wrapper = mount(ImageResultDisplay)

    expect(wrapper.getComponent(UiSelect).exists()).toBe(true)

    await wrapper.get('.result-export-actions__format select').setValue('pdf')
    const downloadAllButton = wrapper
      .findAllComponents(UiButton)
      .find(button => button.text().includes('下载所有图片'))
    expect(downloadAllButton).toBeTruthy()
    await downloadAllButton!.trigger('click')

    expect(exportImportMock.downloadAllImages).toHaveBeenCalledWith('pdf')
  })

  it('groups image and export actions through shared product action rows', () => {
    exportImportMock.state.isDownloading = false
    const imageStore = useImageStore()
    imageStore.addImage('page.png', 'data:image/png;base64,aW1hZ2U=')

    const wrapper = mount(ImageResultDisplay)

    const actionRows = wrapper.findAllComponents(ProductActionRow)
    expect(actionRows.map(row => row.props('ariaLabel'))).toEqual([
      '图片查看操作',
      '翻译结果导出操作',
    ])
    expect(actionRows.every(row => row.props('justify') === 'center')).toBe(true)
    expect(wrapper.find('.image-controls').exists()).toBe(false)
    expect(wrapper.find('.download-buttons').exists()).toBe(false)
  })

  it('uses the typed file-input contract for text import', async () => {
    exportImportMock.state.isDownloading = false
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/result/ExportActions.vue'), 'utf8')
    expect(source).toContain('@files-change="handleImportFile"')
    expect(source).toContain('importFileInput.value?.clear()')
    expect(source).not.toContain('ref<HTMLInputElement')
    expect(source).not.toContain('event.target as HTMLInputElement')
    expect(source).not.toContain('@change="handleImportFile"')
    expect(source).not.toContain('input.value =')

    const imageStore = useImageStore()
    imageStore.addImage('page.png', 'data:image/png;base64,aW1hZ2U=')
    const wrapper = mount(ImageResultDisplay)
    const file = new File(['{}'], 'translation.json', { type: 'application/json' })

    wrapper.getComponent(UiFileInput).vm.$emit('files-change', [file])
    await wrapper.vm.$nextTick()

    expect(exportImportMock.importText).toHaveBeenCalledWith(file)
  })

  it('updates image size through typed range model events', async () => {
    exportImportMock.state.isDownloading = false
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ImageResultDisplay.vue'), 'utf8')
    expect(source).not.toContain('@input="updateImageSize"')
    expect(source).not.toContain('event.target as HTMLInputElement\n  const nextSize')
    expect(source).toContain('@update-image-size="updateImageSize"')

    const toolbarSource = readFileSync(resolve(process.cwd(), 'src/components/translate/result/ResultToolbar.vue'), 'utf8')
    expect(toolbarSource).toContain('@update:model-value="$emit(\'updateImageSize\', $event)"')

    const imageStore = useImageStore()
    imageStore.addImage('page.png', 'data:image/png;base64,aW1hZ2U=')

    const wrapper = mount(ImageResultDisplay)
    const rangeInput = wrapper.getComponent(UiInput)
    rangeInput.vm.$emit('update:modelValue', '175')
    await wrapper.vm.$nextTick()

    expect(wrapper.get('.result-toolbar__image-size-value').text()).toBe('175%')
    rangeInput.vm.$emit('update:modelValue', '500')
    await wrapper.vm.$nextTick()
    expect(wrapper.get('.result-toolbar__image-size-value').text()).toBe('200%')
  })

  it('routes the image-size toolbar control through an inline field primitive', () => {
    exportImportMock.state.isDownloading = false
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/result/ResultToolbar.vue'), 'utf8')
    expect(source).toContain('layout="inline"')
    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('.image-size-control label')

    const imageStore = useImageStore()
    imageStore.addImage('page.png', 'data:image/png;base64,aW1hZ2U=')

    const wrapper = mount(ImageResultDisplay)
    const imageSizeField = wrapper
      .findAllComponents(UiField)
      .find(field => field.props('label') === '图片大小')

    expect(imageSizeField?.props('variant')).toBe('settings')
    expect(imageSizeField?.props('layout')).toBe('inline')
    expect(imageSizeField?.props('controlId')).toBe('imageSize')
  })

  it('does not keep legacy DOM id hooks for result toolbar buttons', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/result/ResultToolbar.vue'), 'utf8')

    for (const legacyId of [
      'id="toggleImageButton"',
      'id="toggleEditModeButton"',
      'id="retranslateFailedButton"',
    ]) {
      expect(source).not.toContain(legacyId)
    }
    expect(source).toContain('control-id="imageSize"')
    expect(source).toContain('id="imageSize"')
  })

  it('does not keep legacy DOM id hooks for result export actions', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/result/ExportActions.vue'), 'utf8')

    for (const legacyId of [
      'id="downloadButton"',
      'id="downloadAllImagesButton"',
      'id="exportTextButton"',
      'id="importTextButton"',
      'id="importTextFileInput"',
    ]) {
      expect(source).not.toContain(legacyId)
    }
    expect(source).toContain('ref="importFileInput"')
    expect(source).toContain('@files-change="handleImportFile"')
  })

  it('delegates detected-text scrolling and list semantics to the product scroll stack', () => {
    exportImportMock.state.isDownloading = false
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ImageResultDisplay.vue'), 'utf8')
    const panelSource = readFileSync(resolve(process.cwd(), 'src/components/translate/DetectedTextPanel.vue'), 'utf8')
    expect(source).toContain('DetectedTextPanel')
    expect(source).not.toContain('<pre class="detected-text-list"')
    expect(panelSource).toContain('aria-labelledby="detectedTextTitle"')
    expect(panelSource).toContain('id="detectedTextTitle"')
    expect(panelSource).not.toContain('id="detectedTextInfo"')
    expect(panelSource).toContain('icon-name="scan-line"')
    expect(panelSource).not.toContain('icon-name="scan-text"')

    const imageStore = useImageStore()
    imageStore.addImage('page.png', 'data:image/png;base64,aW1hZ2U=', {
      originalTexts: ['第一句原文。第二句原文。'],
      bubbleTexts: ['第一句译文。第二句译文。'],
    })

    const wrapper = mount(ImageResultDisplay)
    const textStack = wrapper.getComponent(ProductScrollStack)

    expect(textStack.props()).toMatchObject({
      role: 'list',
      ariaLabel: '检测文本列表',
      gap: 'sm',
      padding: 'none',
    })
    expect(wrapper.find('pre.detected-text-list').exists()).toBe(false)
    expect(wrapper.findAll('[role="listitem"]')).toHaveLength(1)
    expect(wrapper.text()).toContain('第一句原文')
    expect(wrapper.text()).toContain('第一句译文')
  })

  it('does not render a pseudo empty-state node when no image is selected', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ImageResultDisplay.vue'), 'utf8')
    expect(source).not.toContain('empty-state-section')

    const wrapper = mount(ImageResultDisplay)

    expect(wrapper.find('.empty-state-section').exists()).toBe(false)
    expect(wrapper.find('.image-result-display').exists()).toBe(false)
  })

  it('keeps image frame colors on semantic tokens instead of raw values', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ImageResultDisplay.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    expect(styleBlock).not.toMatch(/var\(--color-[a-z0-9-]+,\s*var\(--color-[a-z0-9-]+\)\)/)
    expect(styleBlock).toContain('--shadow-soft')
  })

  it('keeps result toolbar canvas and export actions in explicit child owners', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ImageResultDisplay.vue'), 'utf8')
    const toolbarSource = readFileSync(resolve(process.cwd(), 'src/components/translate/result/ResultToolbar.vue'), 'utf8')
    const canvasSource = readFileSync(resolve(process.cwd(), 'src/components/translate/result/ResultImageCanvas.vue'), 'utf8')
    const detectedTextSource = readFileSync(resolve(process.cwd(), 'src/components/translate/DetectedTextPanel.vue'), 'utf8')
    const exportActionsSource = readFileSync(resolve(process.cwd(), 'src/components/translate/result/ExportActions.vue'), 'utf8')

    expect(source).toContain("import ResultToolbar from '@/components/translate/result/ResultToolbar.vue'")
    expect(source).toContain("import ResultImageCanvas from '@/components/translate/result/ResultImageCanvas.vue'")
    expect(source).toContain("import ExportActions from '@/components/translate/result/ExportActions.vue'")
    expect(source).toContain('data-testid="translation-result-display"')
    expect(source).not.toContain('result-section')
    expect(source).not.toContain('result-card')
    expect(source).not.toContain('content-container')
    expect(source).not.toContain('download-section')
    expect(source).not.toContain('download-all-container')
    expect(source).not.toContain('download-format-selector')
    expect(toolbarSource).toContain('class="result-toolbar__slider"')
    expect(toolbarSource).toContain('class="result-toolbar__image-size-value"')
    expect(toolbarSource).not.toContain('range-slider')
    expect(toolbarSource).not.toContain('class="image-size-value')
    expect(canvasSource).toContain('class="result-image-canvas__image"')
    expect(canvasSource).not.toContain('translated-image')
    expect(detectedTextSource).toContain('class="detected-text-panel__title"')
    expect(detectedTextSource).not.toContain('.detected-text-panel h3')
    expect(exportActionsSource).toContain('class="result-export-actions__progress-label"')
    expect(exportActionsSource).not.toContain('download-progress-label')
  })
})
