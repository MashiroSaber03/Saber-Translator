import { enableAutoUnmount, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { nextTick } from 'vue'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ReaderControls from '@/components/reader/ReaderControls.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiColorSwatchGroup from '@/components/ui/UiColorSwatchGroup.vue'
import UiField from '@/components/ui/UiField.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'

enableAutoUnmount(afterEach)

function mountControls() {
  return mount(ReaderControls, {
    props: {
      hasPrevChapter: true,
      hasNextChapter: true,
      showChapterNav: true,
    },
  })
}

async function requestSettingsPanel(wrapper: ReturnType<typeof mountControls>, requestId = 1) {
  await wrapper.setProps({ settingsRequestId: requestId })
  await nextTick()
}

function readScopedStyle(filePath: string): string {
  const source = readFileSync(resolve(process.cwd(), filePath), 'utf8')
  return source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''
}

function readCssBlock(style: string, selector: string): string {
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return style.match(new RegExp(`${escapedSelector}\\s*\\{([\\s\\S]*?)\\}`))?.[1] ?? ''
}

function nonVariableDeclarations(block: string): string {
  return block
    .split('\n')
    .filter(line => !line.trim().startsWith('--'))
    .join('\n')
}

describe('ReaderControls', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    localStorage.clear()
    document.querySelectorAll('.reader-canvas__stream').forEach(element => element.remove())
    document.documentElement.style.removeProperty('--reader-page-background')
    document.documentElement.style.removeProperty('--reader-image-width')
    document.documentElement.style.removeProperty('--reader-gap')
  })

  it('names icon-only controls and color swatches', async () => {
    const wrapper = mountControls()

    expect(wrapper.get('nav[aria-label="章节导航"]').exists()).toBe(true)
    expect(wrapper.getComponent(UiIconButton).props('label')).toBe('回到顶部')

    await requestSettingsPanel(wrapper)

    const settingsPanel = wrapper.get('.reader-controls__settings-panel')
    expect(settingsPanel.attributes('role')).toBe('dialog')
    expect(settingsPanel.attributes('aria-modal')).toBe('true')
    expect(settingsPanel.attributes('aria-label')).toBe('阅读设置')
    expect(wrapper.get('.reader-controls__close-button').attributes('aria-label')).toBe('关闭阅读设置')
    expect(wrapper.get('.reader-controls__close-button').getComponent(UiIcon).props('name')).toBe('x')
    expect(wrapper.get('.reader-controls__close-button').text()).not.toContain('×')
    const swatches = wrapper.getComponent(UiColorSwatchGroup)
    expect(swatches.props('ariaLabel')).toBe('阅读背景颜色')
    expect(swatches.props('options')).toEqual([
      { value: '#1a1a2e', label: '深蓝' },
      { value: '#ffffff', label: '白色' },
      { value: '#f5f5dc', label: '米色' },
      { value: '#2d2d2d', label: '深灰' },
    ])
    expect(wrapper.find('.reader-controls__bg-option').exists()).toBe(false)
  })

  it('ignores incomplete or invalid stored settings payloads', () => {
    localStorage.setItem('readerSettings', JSON.stringify({
      imageWidth: 10,
      imageGap: 200,
      bgColor: 'not-a-reader-preset',
    }))

    mountControls()

    expect(document.documentElement.style.getPropertyValue('--reader-image-width')).toBe('100%')
    expect(document.documentElement.style.getPropertyValue('--reader-gap')).toBe('8px')
    expect(document.documentElement.style.getPropertyValue('--reader-page-background')).toBe('#1a1a2e')
  })

  it('loads complete current-schema stored settings', () => {
    localStorage.setItem('readerSettings', JSON.stringify({
      readerSettingsSchemaVersion: 1,
      imageWidth: 80,
      imageGap: 12,
      bgColor: '#ffffff',
    }))

    mountControls()

    expect(document.documentElement.style.getPropertyValue('--reader-image-width')).toBe('80%')
    expect(document.documentElement.style.getPropertyValue('--reader-gap')).toBe('12px')
    expect(document.documentElement.style.getPropertyValue('--reader-page-background')).toBe('#ffffff')
  })

  it('tracks and scrolls the virtual page stream instead of the window', async () => {
    const stream = document.createElement('div')
    stream.className = 'reader-canvas__stream'
    Object.defineProperty(stream, 'scrollTop', {
      configurable: true,
      value: 0,
      writable: true,
    })
    Object.defineProperty(stream, 'scrollHeight', {
      configurable: true,
      value: 4000,
    })
    const scrollTo = vi.fn((options: ScrollToOptions) => {
      stream.scrollTop = Number(options.top ?? 0)
      stream.dispatchEvent(new Event('scroll'))
    })
    stream.scrollTo = scrollTo
    document.body.appendChild(stream)

    const wrapper = mountControls()
    await nextTick()

    stream.scrollTop = 800
    stream.dispatchEvent(new Event('scroll'))
    await nextTick()
    expect(wrapper.get('.reader-controls__scroll-top-layer').isVisible()).toBe(true)

    const scrollTopButton = wrapper.findAllComponents(UiIconButton)
      .find(item => item.props('label') === '回到顶部')
    await scrollTopButton!.trigger('click')
    expect(scrollTo).toHaveBeenLastCalledWith({ top: 0, behavior: 'smooth' })

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'End' }))
    expect(scrollTo).toHaveBeenLastCalledWith({ top: 4000, behavior: 'smooth' })

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Home' }))
    expect(scrollTo).toHaveBeenLastCalledWith({ top: 0, behavior: 'smooth' })
  })

  it('renders reader settings fields through the shared settings field primitive', async () => {
    const wrapper = mountControls()

    await requestSettingsPanel(wrapper)

    const fields = wrapper.findAllComponents(UiField)
    expect(fields.map(field => field.props('variant'))).toEqual(['settings', 'settings', 'settings'])
    expect(fields.map(field => field.props('tone'))).toEqual(['inverse', 'inverse', 'inverse'])
    expect(fields.map(field => field.props('label'))).toEqual(['图片宽度', '图片间距', '背景颜色'])
    expect(fields.map(field => field.props('controlId'))).toEqual([
      'imageWidthSlider',
      'imageGapSlider',
      '',
    ])
  })

  it('delegates floating action chrome to the shared button primitive', async () => {
    const wrapper = mountControls()

    const buttonByText = (text: string) => {
      const button = wrapper.findAllComponents(UiButton).find(item => item.text().includes(text))
      expect(button).toBeTruthy()
      return button!
    }

    expect(buttonByText('上一章').props()).toMatchObject({
      variant: 'inverse',
      size: 'md',
    })
    expect(buttonByText('下一章').props()).toMatchObject({
      variant: 'inverse',
      size: 'md',
    })

    await requestSettingsPanel(wrapper)

    const iconButtons = wrapper.findAllComponents(UiIconButton)
    const scrollTopButton = iconButtons.find(item => item.props('label') === '回到顶部')
    const closeButton = iconButtons.find(item => item.classes().includes('reader-controls__close-button'))

    expect(scrollTopButton?.props()).toMatchObject({
      variant: 'primary',
      size: 'xl',
      shape: 'circle',
      elevated: true,
    })
    expect(closeButton?.props()).toMatchObject({
      variant: 'inverse',
      size: 'sm',
      shape: 'circle',
    })

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/reader/ReaderControls.vue'),
      'utf8',
    )
    expect(source).not.toContain('variant="toolbar"')
    expect(source).not.toContain('--ui-button-')

    const style = readScopedStyle('src/components/reader/ReaderControls.vue')
    const localButtonSkinProperties = /^\s*(background|border(?:-radius)?|color|cursor|font-size|height|padding|transition|width)\s*:/m
    for (const selector of [
      '.reader-controls__nav-button',
      '.reader-controls__scroll-top-button',
      '.reader-controls__close-button',
    ]) {
      expect(nonVariableDeclarations(readCssBlock(style, selector))).not.toMatch(localButtonSkinProperties)
    }
  })

  it('does not keep legacy DOM id hooks for reader controls', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/reader/ReaderControls.vue'),
      'utf8',
    )

    for (const legacyId of [
      'id="chapterNav"',
      'id="prevChapterBtn"',
      'id="nextChapterBtn"',
      'id="scrollTopBtn"',
      'id="settingsPanel"',
      'id="imageWidthValue"',
      'id="imageGapValue"',
    ]) {
      expect(source).not.toContain(legacyId)
    }
  })

  it('uses typed model updates for reader range settings', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/reader/ReaderControls.vue'),
      'utf8',
    )

    expect(source).not.toContain('@input="updateImageWidth(Number(($event.target as HTMLInputElement).value))"')
    expect(source).not.toContain('@input="updateImageGap(Number(($event.target as HTMLInputElement).value))"')
    expect(source).toContain('@update:model-value="value => updateImageWidth(Number(value))"')
    expect(source).toContain('@update:model-value="value => updateImageGap(Number(value))"')
  })

  it('opens settings from a typed request prop instead of exposed instance methods', () => {
    const controlsSource = readFileSync(
      resolve(process.cwd(), 'src/components/reader/ReaderControls.vue'),
      'utf8',
    )
    const viewSource = readFileSync(
      resolve(process.cwd(), 'src/views/ReaderView.vue'),
      'utf8',
    )

    expect(controlsSource).not.toContain('defineExpose')
    expect(controlsSource).toContain('settingsRequestId')
    expect(viewSource).not.toContain('readerControlsRef')
    expect(viewSource).toContain(':settings-request-id="settingsRequestId"')
  })

  it('uses the shared dialog lifecycle and restores focus after closing settings', async () => {
    const trigger = document.createElement('button')
    trigger.textContent = 'Open reader settings'
    document.body.appendChild(trigger)
    trigger.focus()

    const wrapper = mount(ReaderControls, {
      attachTo: document.body,
      props: {
        hasPrevChapter: true,
        hasNextChapter: true,
        showChapterNav: true,
      },
    })
    await requestSettingsPanel(wrapper)

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/reader/ReaderControls.vue'),
      'utf8',
    )
    expect(source).toContain('useDialogLifecycle')
    expect(source).not.toContain("case 'Escape':")
    expect(document.activeElement).toBe(wrapper.get('.reader-controls__settings-content').element)

    await wrapper.get('.reader-controls__close-button').trigger('click')
    await nextTick()
    expect(document.activeElement).toBe(trigger)
  })

  it('maps controls style owner colors through semantic tokens while keeping preset values as data', () => {
    const style = readScopedStyle('src/components/reader/ReaderControls.vue')

    expect(style).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    expect(style).toContain('--reader-controls-chapter-nav-start: color-mix')
    expect(style).toContain('--reader-controls-settings-panel-background: var(--color-surface-inverse-raised)')
  })

  it('keeps settings panel text hooks explicit instead of element descendant selectors', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/reader/ReaderControls.vue'),
      'utf8',
    )

    expect(source).toContain('reader-controls__settings-title')
    expect(source).toContain('reader-controls__setting-value')
    expect(source).not.toMatch(/\.reader-controls__settings-header\s+h3/)
    expect(source).not.toMatch(/\.reader-controls__setting-control\s+span/)
    expect(source).not.toContain('--color-text-default')
  })
})
