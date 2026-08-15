import { enableAutoUnmount, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { nextTick } from 'vue'
import { afterEach, describe, expect, it } from 'vitest'

import ReferenceImageSelector from './ReferenceImageSelector.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductThumbnailGrid from '@/components/product/ProductThumbnailGrid.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'

enableAutoUnmount(afterEach)

afterEach(() => {
  document.body.innerHTML = ''
})

async function clickConfirmButton(wrapper: ReturnType<typeof mount>): Promise<void> {
  await nextTick()
  const dialogActions = wrapper
    .findAllComponents(ProductActionRow)
    .find(row => row.props('ariaLabel') === '参考图选择器操作')
  expect(dialogActions).toBeTruthy()
  const buttons = dialogActions?.findAll('button') ?? []
  expect(buttons.length).toBeGreaterThanOrEqual(2)
  await buttons[buttons.length - 1]!.trigger('click')
  await nextTick()
}

describe('ReferenceImageSelector', () => {
  it('auto-selects and emits reference tokens instead of raw paths', async () => {
    const wrapper = mount(ReferenceImageSelector, {
      attachTo: document.body,
      props: {
        visible: true,
        mode: 'script',
        maxCount: 1,
        originalImages: [
          {
            page_number: 1,
            path: '/tmp/page-1.png',
            has_image: true,
            token: 'original:1',
          },
        ],
        continuationImages: [],
        characterForms: [],
        initialSelection: [],
      },
    })

    await clickConfirmButton(wrapper)

    expect(wrapper.emitted('confirm')?.[0]).toEqual([['original:1']])
  })

  it('prefers the latest real images instead of placeholder continuation pages', async () => {
    const wrapper = mount(ReferenceImageSelector, {
      attachTo: document.body,
      props: {
        visible: true,
        mode: 'image',
        maxCount: 2,
        originalImages: [
          {
            page_number: 185,
            path: '/tmp/page-185.png',
            has_image: true,
            token: 'original:185',
          },
          {
            page_number: 186,
            path: '/tmp/page-186.png',
            has_image: true,
            token: 'original:186',
          },
        ],
        continuationImages: [
          {
            page_number: 187,
            path: '',
            has_image: false,
            token: 'continuation:1',
            is_placeholder: true,
          },
          {
            page_number: 188,
            path: '',
            has_image: false,
            token: 'continuation:2',
            is_placeholder: true,
          },
        ],
        characterForms: [],
        initialSelection: [],
      },
    })

    await clickConfirmButton(wrapper)

    expect(wrapper.emitted('confirm')?.[0]).toEqual([['original:185', 'original:186']])
  })

  it('uses the product thumbnail grid for selectable manga thumbnails', async () => {
    const wrapper = mount(ReferenceImageSelector, {
      attachTo: document.body,
      props: {
        visible: true,
        mode: 'script',
        maxCount: 2,
        originalImages: [
          {
            page_number: 1,
            path: '/tmp/page-1.png',
            has_image: true,
            token: 'original:1',
          },
          {
            page_number: 2,
            path: '/tmp/page-2.png',
            has_image: true,
            token: 'original:2',
          },
        ],
        continuationImages: [],
        characterForms: [],
        initialSelection: [],
      },
    })

    await nextTick()
    const grid = wrapper
      .findAllComponents(ProductThumbnailGrid)
      .find(component => component.props('ariaLabel') === '漫画参考图选择')
    if (!grid) throw new Error('Missing manga thumbnail grid')
    expect(grid.props('ariaLabel')).toBe('漫画参考图选择')
    expect(grid.props('items')).toHaveLength(2)

    const firstThumbnail = document.body.querySelector<HTMLButtonElement>('button[aria-label="取消选择原作第1页参考图"]')
    expect(firstThumbnail).not.toBeNull()
    expect(firstThumbnail?.getAttribute('aria-pressed')).toBe('true')

    firstThumbnail?.click()
    await clickConfirmButton(wrapper)

    expect(wrapper.emitted('confirm')?.[0]).toEqual([['original:2']])
  })

  it('keeps scrolling and columns owned by the virtual thumbnail grid', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/continuation/ReferenceImageSelector.vue'),
      'utf8'
    )
    const scrollBlock = source.match(/\.reference-image-selector__scroll\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''
    const productGridBlock = source.match(/\.reference-image-selector__thumbnail-grid\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''

    expect(source).not.toContain('class="thumbnails-grid"')
    expect(source).not.toContain('reference-selector-scroll')
    expect(source).not.toContain('reference-thumbnail-grid')
    expect(source).not.toMatch(/\.thumbnails-grid\s*\{[\s\S]*?grid-template-columns/)
    expect(scrollBlock).toContain('min-height: 0')
    expect(scrollBlock).not.toMatch(/display:\s*grid|grid-template-columns|justify-content/)
    expect(source).toContain("import VirtualThumbnailGrid from '@/components/virtual/VirtualThumbnailGrid.vue'")
    expect(source).toContain(':min-item-width="110"')
    expect(source).toContain(':max-height="560"')
    expect(productGridBlock).toContain('--product-thumbnail-grid-min-size: 110px')
  })

  it('renders character reference forms through the product thumbnail grid as static cards', async () => {
    const wrapper = mount(ReferenceImageSelector, {
      attachTo: document.body,
      props: {
        visible: true,
        mode: 'image',
        maxCount: 2,
        originalImages: [],
        continuationImages: [],
        characterForms: [
          {
            character_name: 'Saber',
            form_id: 'casual',
            form_name: '常服',
            path: '/tmp/saber-casual.png',
            has_image: true,
            token: 'form:saber:casual',
          },
        ],
        initialSelection: [],
      },
    })

    await nextTick()

    const characterGrid = wrapper
      .findAllComponents(ProductThumbnailGrid)
      .find(component => component.props('ariaLabel') === '角色档案参考图')
    if (!characterGrid) throw new Error('Missing character form thumbnail grid')
    expect(characterGrid.props('items')).toEqual([
      expect.objectContaining({
        id: 'form:saber:casual',
        interactive: false,
        label: 'Saber - 常服',
      }),
    ])
    expect(wrapper.find('.character-thumbnail').exists()).toBe(false)
  })

  it('uses an explicit icon-only close action', async () => {
    const wrapper = mount(ReferenceImageSelector, {
      attachTo: document.body,
      props: {
        visible: true,
        mode: 'script',
        maxCount: 1,
        originalImages: [],
        continuationImages: [],
        characterForms: [],
        initialSelection: [],
      },
    })

    await nextTick()

    const closeButton = document.body.querySelector<HTMLButtonElement>('button[aria-label="关闭参考图选择器"]')
    expect(closeButton).not.toBeNull()
    expect(closeButton?.textContent?.trim()).toBe('')
    const closeAction = wrapper.getComponent(UiIconButton)
    expect(closeAction.props('label')).toBe('关闭参考图选择器')
    expect(closeAction.props('title')).toBe('关闭')
  })

  it('uses product action rows for batch and dialog actions', async () => {
    const wrapper = mount(ReferenceImageSelector, {
      attachTo: document.body,
      props: {
        visible: true,
        mode: 'script',
        maxCount: 1,
        originalImages: [],
        continuationImages: [],
        characterForms: [],
        initialSelection: [],
      },
    })

    await nextTick()

    const rows = wrapper.findAllComponents(ProductActionRow)
    expect(rows.some(row => row.props('ariaLabel') === '参考图批量操作')).toBe(true)

    const dialogActions = rows.find(row => row.props('ariaLabel') === '参考图选择器操作')
    expect(dialogActions?.props('variant')).toBe('dialog')
  })

  it('uses semantic tokens instead of raw owner color values', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/continuation/ReferenceImageSelector.vue'),
      'utf8'
    )
    const ownerTokenBlock = source.match(/\.reference-image-selector\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''

    expect(ownerTokenBlock).not.toMatch(/#[0-9a-f]{3,8}\b/i)
  })

  it('keeps modal container visuals on typed BaseModal props instead of a global style entry', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/continuation/ReferenceImageSelector.vue'),
      'utf8'
    )

    expect(source).not.toContain('ReferenceImageSelector.global.styles.css')
    expect(source).toContain('frame-variant="outlined"')
    expect(source).not.toContain('border="1px solid var(--color-border-default)"')
    expect(source).not.toContain('border-radius="18px"')
    expect(source).not.toContain('box-shadow="0 24px 64px var(--shadow-medium)"')
  })

  it('lets the modal header wrap without fixed nowrap pressure', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/continuation/ReferenceImageSelector.vue'),
      'utf8'
    )
    const headerBlock = source.match(/\.reference-image-selector__header\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''
    const titleBlock = source.match(/\.reference-image-selector__title\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''
    const batchActionsBlock = source.match(/\.reference-image-selector__batch-actions\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''
    const dialogActionsBlock = source.match(/\.reference-image-selector__dialog-actions\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''

    expect(headerBlock).toContain('flex-wrap: wrap')
    expect(titleBlock).toContain('min-width: 0')
    expect(titleBlock).toContain('overflow-wrap: anywhere')
    expect(titleBlock).not.toContain('white-space: nowrap')
    expect(batchActionsBlock).toContain('min-width: 0')
    expect(dialogActionsBlock).toContain('min-width: 0')
  })

  it('uses ReferenceImageSelector owner hooks instead of generic modal-local classes', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/continuation/ReferenceImageSelector.vue'),
      'utf8'
    )
    const classTokens = [...source.matchAll(/class="([^"]+)"/g)]
      .flatMap(match => match[1]!.split(/\s+/).filter(Boolean))

    expect(source).toContain('reference-image-selector')
    expect(source).toContain('reference-image-selector__header')
    expect(source).toContain('reference-image-selector__batch-actions')
    expect(source).toContain('reference-image-selector__dialog-actions')
    expect(source).toContain('reference-image-selector__thumbnail-grid')

    for (const legacyClass of [
      'reference-selector-content',
      'modal-header',
      'header-actions',
      'header-right',
      'close-btn',
      'character-section',
      'section-label',
      'section-hint',
      'character-thumbnail-grid',
      'manga-section',
      'reference-selector-scroll',
      'reference-thumbnail-grid',
    ]) {
      expect(classTokens).not.toContain(legacyClass)
    }
  })
})
