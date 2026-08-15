import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import ImageResultDisplay from '@/components/translate/ImageResultDisplay.vue'
import DetectedTextPanel from '@/components/translate/DetectedTextPanel.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductScrollStack from '@/components/product/ProductScrollStack.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiField from '@/components/ui/UiField.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { useImageStore } from '@/stores/imageStore'
import { addTestImage } from '../helpers/imageFixtures'
import { useSettingsStore } from '@/stores/settings'
import { createBubbleState } from '@/utils/bubbleFactory'

const exportImportMock = vi.hoisted(() => ({
  state: {
    isDownloading: true,
    isImporting: false,
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
    isImporting: { value: exportImportMock.state.isImporting },
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
    exportImportMock.state.isImporting = false
    exportImportMock.state.downloadProgressText = '正在打包下载'
    exportImportMock.state.downloadProgress = 42
    exportImportMock.downloadCurrentImage.mockReset()
    exportImportMock.downloadAllImages.mockReset()
    exportImportMock.exportText.mockReset()
    exportImportMock.importText.mockReset()
  })

  it('renders download progress through the shared progress primitive', () => {
    const imageStore = useImageStore()
    addTestImage(imageStore, 'page.png', 'data:image/png;base64,aW1hZ2U=')

    const wrapper = mount(ImageResultDisplay)

    const progress = wrapper.getComponent(UiProgressBar)
    expect(progress.props('value')).toBe(42)
    expect(progress.text()).toContain('正在打包下载')
    expect(wrapper.find('.progress').exists()).toBe(false)
    expect(wrapper.findAllComponents(UiButton).filter(button => (
      ['下载当前图片', '下载所有图片', '导出文本', '导入文本'].includes(button.text())
    )).every(button => button.props('disabled'))).toBe(true)
  })

  it('lets the image-size control enlarge the image beyond its frame width', async () => {
    exportImportMock.state.isDownloading = false
    const imageStore = useImageStore()
    addTestImage(imageStore, 'page.png', '/api/v2/assets/source-1')
    const wrapper = mount(ImageResultDisplay)

    const slider = wrapper.getComponent(UiInput)
    slider.vm.$emit('update:modelValue', 200)
    await wrapper.vm.$nextTick()

    expect(wrapper.get('.result-image-canvas__image-layer').attributes('style')).toContain('width: 200%')
    const canvasSource = readFileSync(
      resolve(process.cwd(), 'src/components/translate/result/ResultImageCanvas.vue'),
      'utf8',
    )
    const layerStyles = canvasSource.match(/\.result-image-canvas__image-layer\s*\{([\s\S]*?)\n\}/)?.[1] ?? ''
    expect(layerStyles).toContain('flex: 0 0 auto')
    expect(layerStyles).not.toContain('max-width: 100%')
  })

  it('preserves source text and leaves visual wrapping to CSS', () => {
    const text = '这是一段不会被组件擅自插入换行符的长文本。'.repeat(5)
    const wrapper = mount(DetectedTextPanel, {
      props: { items: [{ original: text, translated: text }] },
    })

    expect(wrapper.get('.detected-text-panel__original').text()).toBe(text)
    expect(wrapper.get('.detected-text-panel__translated').text()).toBe(text)
  })

  it('shows a clean asset as the final remove-text result and can toggle back to source', async () => {
    exportImportMock.state.isDownloading = false
    const imageStore = useImageStore()
    addTestImage(imageStore, 'page.png', '/api/v2/assets/source-1', {
      cleanAssetUrl: '/api/v2/assets/clean-1',
    })

    const wrapper = mount(ImageResultDisplay)
    const resultImage = wrapper.get('.result-image-canvas__image')

    expect(resultImage.attributes('src')).toBe('/api/v2/assets/clean-1')
    expect(resultImage.attributes('alt')).toBe('消字图：page.png')
    const toggle = wrapper
      .findAllComponents(UiButton)
      .find(button => button.text().includes('查看原图'))
    expect(toggle).toBeTruthy()

    await toggle!.trigger('click')

    expect(resultImage.attributes('src')).toBe('/api/v2/assets/source-1')
    expect(resultImage.attributes('alt')).toBe('原图：page.png')
    expect(wrapper.text()).toContain('查看消字图')
  })

  it('shows detection boxes only when the saved debug setting is enabled', async () => {
    exportImportMock.state.isDownloading = false
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    addTestImage(imageStore, 'page.png', '/api/v2/assets/source-1', {
      width: 1000,
      height: 1500,
      bubbleStates: [{
        ...createBubbleState(),
        backendBubbleId: 'bubble-1',
        coords: [100, 200, 400, 500],
      }],
    })

    const wrapper = mount(ImageResultDisplay)
    expect(wrapper.find('[data-testid="detection-debug-overlay"]').exists()).toBe(false)

    settingsStore.settings.showDetectionDebug = true
    await wrapper.vm.$nextTick()

    const overlay = wrapper.get('[data-testid="detection-debug-overlay"]')
    expect(overlay.attributes('viewBox')).toBe('0 0 1000 1500')
    expect(overlay.get('rect').attributes()).toMatchObject({
      x: '100',
      y: '200',
      width: '300',
      height: '300',
    })

    settingsStore.settings.showDetectionDebug = false
    await wrapper.vm.$nextTick()
    expect(wrapper.find('[data-testid="detection-debug-overlay"]').exists()).toBe(false)
  })

  it('uses the product select primitive for fixed download formats', async () => {
    exportImportMock.state.isDownloading = false
    const imageStore = useImageStore()
    addTestImage(imageStore, 'page.png', 'data:image/png;base64,aW1hZ2U=')

    const wrapper = mount(ImageResultDisplay)

    const formatSelect = wrapper.getComponent(UiSelect)
    expect(formatSelect.exists()).toBe(true)

    formatSelect.vm.$emit('update:modelValue', 'pdf')
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
    addTestImage(imageStore, 'page.png', 'data:image/png;base64,aW1hZ2U=')

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
    addTestImage(imageStore, 'page.png', 'data:image/png;base64,aW1hZ2U=')
    const wrapper = mount(ImageResultDisplay)
    const file = new File(['{}'], 'translation.json', { type: 'application/json' })

    wrapper.getComponent(UiFileInput).vm.$emit('files-change', [file])
    await wrapper.vm.$nextTick()

    expect(exportImportMock.importText).toHaveBeenCalledWith(file)
  })

  it('prevents overlapping backend export submissions while one export is pending', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/result/ExportActions.vue'), 'utf8')

    expect(source).toContain(':disabled="!hasImages || isDownloading || isImporting"')
    expect(source).toContain(':disabled="isDownloading || isImporting"')
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
    addTestImage(imageStore, 'page.png', 'data:image/png;base64,aW1hZ2U=')

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
    addTestImage(imageStore, 'page.png', 'data:image/png;base64,aW1hZ2U=')

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
    addTestImage(imageStore, 'page.png', '/api/v2/assets/source-1', {
      bubbleStates: [{
        ...createBubbleState(),
        originalText: '第一句原文。第二句原文。',
        translatedText: '第一句译文。第二句译文。',
      }],
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
