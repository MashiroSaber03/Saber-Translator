import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import EditToolbar from '@/components/edit/EditToolbar.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'

function createToolbarProps() {
  return {
    currentImageIndex: 0,
    imageCount: 3,
    canGoPrevious: false,
    canGoNext: true,
    showThumbnails: false,
    hasBubbles: true,
    selectedBubbleIndex: 0,
    bubbleCount: 2,
    layoutMode: 'horizontal' as const,
    syncEnabled: false,
    scale: 1,
    isDrawingMode: false,
    hasSelection: true,
    brushMode: null,
    brushSize: 24,
    mouseX: 10,
    mouseY: 20,
    isProcessing: true,
    progressText: '处理中',
    progressCurrent: 1,
    progressTotal: 4,
    isRepairLoading: false,
  }
}

describe('EditToolbar accessibility', () => {
  it('uses an explicit button for the image indicator thumbnail toggle', async () => {
    const wrapper = mount(EditToolbar, {
      props: createToolbarProps(),
    })

    const indicator = wrapper.get('.edit-toolbar__image-indicator')
    expect(indicator.element.tagName).toBe('BUTTON')
    expect(indicator.attributes('aria-label')).toBe('显示或隐藏缩略图')

    await indicator.trigger('click')
    expect(wrapper.emitted('toggle-thumbnails')).toHaveLength(1)
  })

  it('exposes edit processing progress through progressbar semantics', () => {
    const wrapper = mount(EditToolbar, {
      props: createToolbarProps(),
    })

    const sharedProgress = wrapper.getComponent(UiProgressBar)
    expect(sharedProgress.props()).toEqual(expect.objectContaining({
      value: 1,
      max: 4,
      label: '编辑处理进度',
      tone: 'success',
      size: 'sm',
      animated: true,
    }))

    const progressbar = wrapper.get('[role="progressbar"]')
    expect(progressbar.attributes('aria-valuemin')).toBe('0')
    expect(progressbar.attributes('aria-valuemax')).toBe('4')
    expect(progressbar.attributes('aria-valuenow')).toBe('1')
    expect(progressbar.attributes('aria-label')).toBe('编辑处理进度')
  })

  it('renders zoom controls through shared icons instead of text symbols', () => {
    const wrapper = mount(EditToolbar, {
      props: createToolbarProps(),
    })

    const zoomIn = wrapper.get('button[title="放大 (+)"]')
    const zoomOut = wrapper.get('button[title="缩小 (-)"]')

    expect(zoomIn.getComponent(UiIcon).props('name')).toBe('plus')
    expect(zoomOut.getComponent(UiIcon).props('name')).toBe('minus')
    expect(zoomIn.text()).not.toContain('+')
    expect(zoomOut.text()).not.toContain('−')
  })

  it('gives icon-only toolbar buttons explicit accessible names', () => {
    const wrapper = mount(EditToolbar, {
      props: createToolbarProps(),
    })

    const unlabeledIconButtons = wrapper
      .findAll('button')
      .filter((button) => button.text().trim() === '')
      .filter((button) => !button.attributes('aria-label'))
      .map((button) => button.attributes('title') || button.classes().join('.'))

    expect(unlabeledIconButtons).toEqual([])
    expect(wrapper.get('button[title="上一张图片 (A)"]').attributes('aria-label')).toBe('上一张图片')
    expect(wrapper.get('button[title="适应屏幕 (双击)"]').attributes('aria-label')).toBe('适应屏幕')
  })

  it('renders navigation and view icon-only tools through UiIconButton', () => {
    const wrapper = mount(EditToolbar, {
      props: createToolbarProps(),
    })
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditToolbar.vue'),
      'utf8',
    )

    expect(source).toContain("import UiIconButton from '@/components/ui/UiIconButton.vue'")
    expect(wrapper.findAllComponents(UiIconButton).map(button => button.props('label'))).toEqual([
      '上一张图片',
      '下一张图片',
      '显示或隐藏缩略图',
      '上一个气泡',
      '下一个气泡',
      '切换布局',
      '切换视图模式',
      '同步缩放和拖动',
      '适应屏幕',
      '放大',
      '缩小',
      '原始大小',
    ])
    expect(wrapper.get('button[title="上一张图片 (A)"]').attributes('aria-label')).toBe('上一张图片')
    expect(wrapper.get('button[title="放大 (+)"]').attributes('aria-label')).toBe('放大')
    expect(source).not.toMatch(/<UiButton[\s\S]{0,180}class="nav-btn"/)
    expect(source).not.toMatch(/<UiButton[\s\S]{0,180}class="view-control-btn/)
  })

  it('lets UiIconButton own icon-only toolbar chrome', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditToolbar.vue'),
      'utf8',
    )

    expect(source).toContain('--ui-icon-button-active-background: var(--edit-toolbar-chip-active-background)')
    expect(source).not.toMatch(/\.(thumb-toggle-btn|view-control-btn)(?:[:.\s,{])/)
    expect(source).not.toMatch(/\.image-navigator \.nav-btn/)
    expect(source).not.toMatch(/\.bubble-navigator \.nav-btn/)
  })

  it('exposes pressed state for toggle-style toolbar controls', () => {
    const wrapper = mount(EditToolbar, {
      props: {
        ...createToolbarProps(),
        showThumbnails: true,
        syncEnabled: true,
        isDrawingMode: true,
        brushMode: 'repair',
      },
    })

    expect(wrapper.get('.edit-toolbar__image-indicator').attributes('aria-pressed')).toBe('true')
    expect(wrapper.get('button[title="显示/隐藏缩略图"]').attributes('aria-pressed')).toBe('true')
    expect(wrapper.get('button[title="同步缩放/拖动"]').attributes('aria-pressed')).toBe('true')
    expect(wrapper.get('button[title="添加气泡框（或中键拖拽绘制）"]').attributes('aria-pressed')).toBe('true')
    expect(wrapper.get('button[title="修复笔刷 (按住R+左键拖拽)"]').attributes('aria-pressed')).toBe('true')
    expect(wrapper.get('button[title="还原笔刷 (按住U+左键拖拽)"]').attributes('aria-pressed')).toBe('false')
  })

  it('does not keep legacy DOM id hooks for edit toolbar counters', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditToolbar.vue'),
      'utf8',
    )

    for (const legacyId of [
      'id="prevBubbleBtn"',
      'id="currentBubbleNum"',
      'id="totalBubbleNum"',
      'id="nextBubbleBtn"',
      'id="zoomLevel"',
    ]) {
      expect(source).not.toContain(legacyId)
    }
  })

  it('connects the keyboard help trigger to its tooltip content', () => {
    const wrapper = mount(EditToolbar, {
      props: createToolbarProps(),
    })

    const trigger = wrapper.get('button[title="快捷键操作帮助"]')
    const tooltipId = trigger.attributes('aria-describedby')
    expect(tooltipId).toBe('edit-toolbar-help-tooltip')

    const tooltip = wrapper.get(`#${tooltipId}`)
    expect(tooltip.attributes('role')).toBe('tooltip')
    expect(tooltip.text()).toContain('鼠标操作')
    expect(tooltip.text()).toContain('快捷键')
  })

  it('delegates keyboard help tooltip markup to its own toolbar help owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditToolbar.vue'),
      'utf8',
    )

    expect(source).toContain("import EditToolbarHelp from './EditToolbarHelp.vue'")
    expect(source).toContain('<EditToolbarHelp />')
    expect(source).not.toContain('help-tooltip-container')
    expect(source).not.toContain('help-tooltip-popup')
    expect(source).not.toContain('--edit-toolbar-help-trigger')
  })

  it('renders workflow actions with standard button variants', () => {
    const wrapper = mount(EditToolbar, {
      props: createToolbarProps(),
    })

    const buttons = wrapper.findAllComponents(UiButton)
    const exitButton = buttons.find((button) => button.text() === '退出编辑')
    const applyButton = buttons.find((button) => button.text() === '应用并下一张')

    expect(exitButton?.props()).toMatchObject({
      variant: 'inverse',
    })
    expect(exitButton?.classes()).not.toContain('action-secondary')
    expect(applyButton?.props()).toMatchObject({
      variant: 'primary',
      tone: 'success',
    })
    expect(applyButton?.classes()).not.toContain('action-primary')

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditToolbar.vue'),
      'utf8',
    )
    expect(source).not.toContain('--ui-button-')
  })

  it('lets toolbar rows and control groups wrap inside narrow edit workspaces', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditToolbar.vue'),
      'utf8',
    )

    expect(source).toMatch(/\.edit-toolbar__row\s*\{[\s\S]*flex-wrap:\s*wrap/)
    expect(source).toMatch(/\.edit-toolbar__spacer\s*\{[\s\S]*min-width:\s*0/)
    expect(source).toMatch(/\.edit-toolbar__annotation-tools\s*\{[\s\S]*flex-wrap:\s*wrap/)
    expect(source).toMatch(/\.edit-toolbar__view-controls\s*\{[\s\S]*flex-wrap:\s*wrap/)
  })

  it('uses edit-toolbar owner hooks instead of generic local toolbar classes', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditToolbar.vue'),
      'utf8',
    )

    expect(source).toContain('class="edit-toolbar"')
    expect(source).toContain('edit-toolbar__row')
    expect(source).toContain('edit-toolbar__image-navigator')
    expect(source).toContain('edit-toolbar__bubble-navigator')
    expect(source).toContain('edit-toolbar__view-controls')
    expect(source).toContain('edit-toolbar__annotation-tools')
    expect(source).toContain('edit-toolbar__progress')
    expect(source).toContain('edit-toolbar__quick-actions')
    expect(source).not.toContain('edit-toolbar-wrapper')
    expect(source).not.toMatch(/class="[^"]*\b(?:toolbar-row-1|toolbar-row-2|image-navigator|nav-btn|thumb-toggle-btn|toolbar-divider|bubble-navigator|bubble-indicator|view-controls|view-control-btn|layout-toggle-btn|view-mode-btn|sync-toggle-btn|zoom-level|toolbar-spacer|annotation-tools|annotation-btn|detect-btn|primary-action-btn|brush-btn|brush-size-indicator|brush-cursor|brush-mode-hint-layer|brush-mode-hint|edit-progress-container|edit-progress-info|edit-progress-text|edit-progress-count|edit-progress-bar|quick-actions)\b/)
    expect(source).not.toMatch(/\.(?:toolbar-row-1|toolbar-row-2|image-navigator|image-indicator|nav-btn|thumb-toggle-btn|toolbar-divider|bubble-navigator|bubble-indicator|view-controls|view-control-btn|layout-toggle-btn|view-mode-btn|sync-toggle-btn|zoom-level|toolbar-spacer|annotation-tools|annotation-btn|detect-btn|primary-action-btn|brush-btn|brush-size-indicator|brush-cursor|brush-mode-hint-layer|brush-mode-hint|edit-progress-container|edit-progress-info|edit-progress-text|edit-progress-count|edit-progress-bar|quick-actions)\b/)
  })

  it('uses explicit edit-toolbar hooks for counter values and action labels', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditToolbar.vue'),
      'utf8',
    )

    expect(source).toContain('edit-toolbar__image-indicator-value')
    expect(source).toContain('edit-toolbar__bubble-indicator-value')
    expect(source).toContain('edit-toolbar__annotation-icon')
    expect(source).toContain('edit-toolbar__annotation-label')
    expect(source).not.toContain('.edit-toolbar__image-indicator span')
    expect(source).not.toContain('.edit-toolbar__bubble-indicator span')
    expect(source).not.toContain('.edit-toolbar__annotation-action svg')
    expect(source).not.toContain('.edit-toolbar__annotation-action span')
  })

  it('does not assert shared button primitives through internal class names', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/editToolbarAccessibility.spec.ts'), 'utf8')
    const buttonClassPrefix = 'ui-' + 'button--'
    const iconButtonClassPrefix = 'ui-' + 'icon-button--'

    expect(source).not.toContain(buttonClassPrefix)
    expect(source).not.toContain(iconButtonClassPrefix)
  })

  it('maps toolbar owner colors through semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditToolbar.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    expect(source).not.toMatch(/<svg[\s>]/)
    expect(source).toContain('--edit-toolbar-shell-start: var(--color-surface-inverse-panel)')
    expect(source).toContain('--edit-toolbar-status-accent: var(--color-action-success-bright)')
  })

  it('uses numeric font weights for toolbar counters', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditToolbar.vue'),
      'utf8',
    )
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/font-weight:\s*(bold|normal)\b/)
  })
})
