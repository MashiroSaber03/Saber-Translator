import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import ThumbnailSidebar from '@/components/translate/ThumbnailSidebar.vue'
import ProductBreadcrumbTrail from '@/components/product/ProductBreadcrumbTrail.vue'
import ProductFolderCard from '@/components/product/ProductFolderCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import ProductThumbnailGrid from '@/components/product/ProductThumbnailGrid.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { useImageStore } from '@/stores/imageStore'
import type { ImageData } from '@/types/image'

function createImage(id: string, fileName: string, folderPath = ''): ImageData {
  return {
    id,
    fileName,
    folderPath,
    width: 100,
    height: 100,
    sourceAssetUrl: `/api/v2/assets/${id}`,
    thumbnailSourceUrl: `/api/v2/assets/${id}-thumbnail`,
    translatedAssetUrl: null,
    cleanAssetUrl: null,
    bubbleStates: null,
    translationStatus: 'pending',
    hasUnsavedChanges: false,
    fontSize: 18,
    fontFamily: 'Arial',
    textColor: '#000000',
    strokeColor: '#ffffff',
    strokeWidth: 0,
    strokeEnabled: false,
    fillColor: '#ffffff',
    layoutDirection: 'horizontal',
    lineSpacing: 1,
    inlineAlign: 'center',
    blockAlign: 'center',
    autoFontSize: false,
    useAutoTextColor: false,
  } as ImageData
}

function mountSidebar(images: ImageData[]) {
  setActivePinia(createPinia())
  const imageStore = useImageStore()
  imageStore.images = images
  imageStore.currentImageIndex = images.length > 0 ? 0 : -1

  return mount(ThumbnailSidebar)
}

describe('ThumbnailSidebar', () => {
  beforeEach(() => {
    HTMLElement.prototype.scrollTo = vi.fn()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders flat image thumbnails through the product thumbnail grid', async () => {
    const wrapper = mountSidebar([
      createImage('page-1', '001.png'),
      createImage('page-2', '002.png'),
    ])

    const thumbnailGrid = wrapper.getComponent(ProductThumbnailGrid)
    expect(thumbnailGrid.props('ariaLabel')).toBe('图片缩略图导航')
    expect(thumbnailGrid.props('items')).toMatchObject([
      {
        id: 0,
        src: '/api/v2/assets/page-1-thumbnail',
        alt: '001.png',
        label: '1',
        selected: true,
        ariaLabel: '选择图片 1: 001.png',
      },
      {
        id: 1,
        src: '/api/v2/assets/page-2-thumbnail',
        alt: '002.png',
        label: '2',
        selected: false,
        ariaLabel: '选择图片 2: 002.png',
      },
    ])
    expect(wrapper.find('.thumbnail-sidebar__item').exists()).toBe(false)

    thumbnailGrid.vm.$emit('select', 1)

    expect(wrapper.emitted('select')?.[0]).toEqual([1])
  })

  it('keeps folder navigation local and renders tree thumbnails through the product grid', async () => {
    const wrapper = mountSidebar([
      createImage('page-1', '001.png', 'chapter-a'),
      createImage('page-2', '002.png', 'chapter-a'),
    ])

    const folderCard = wrapper.getComponent(ProductFolderCard)
    expect(folderCard.props()).toMatchObject({
      ariaLabel: '打开文件夹 chapter-a',
      count: 2,
      countId: 'chapter-a',
      folderName: 'chapter-a',
    })
    expect(wrapper.find('.folder-item.ui-button').exists()).toBe(false)

    folderCard.vm.$emit('select')
    await wrapper.vm.$nextTick()

    const backButton = wrapper.get('.thumbnail-sidebar__back-button').getComponent(UiButton)
    expect(backButton.props()).toMatchObject({
      variant: 'secondary',
      size: 'sm',
      tone: 'primary',
      block: true,
    })

    const breadcrumbTrail = wrapper.getComponent(ProductBreadcrumbTrail)
    expect(breadcrumbTrail.props('items')).toMatchObject([
      { path: '', name: '根目录' },
      { path: 'chapter-a', name: 'chapter-a' },
    ])
    breadcrumbTrail.vm.$emit('select', '')

    const thumbnailGrid = wrapper.getComponent(ProductThumbnailGrid)
    expect(thumbnailGrid.props('items')).toMatchObject([
      {
        id: 0,
        label: '1',
        selected: true,
      },
      {
        id: 1,
        label: '2',
        selected: false,
      },
    ])

    thumbnailGrid.vm.$emit('select', 1)
    expect(wrapper.emitted('select')?.[0]).toEqual([1])
  })

  it('keeps folder rows on neutral product card and chip semantics', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ThumbnailSidebar.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    expect(styleBlock).not.toContain('--color-status-warning')
    expect(styleBlock).not.toContain('color-mix(in srgb')
    expect(source).toContain('ProductFolderCard')
    expect(source).toContain(':count-id="subfolder.path"')
    expect(source).not.toContain('ProductRecordCard')
    expect(source).not.toContain('ProductChipList')
    expect(source).not.toContain('folder-count')
    expect(source).not.toContain('folderImageCountChips')
    expect(source).not.toContain('class="folder-info"')
    expect(source).not.toContain('class="folder-name"')
    expect(styleBlock).toContain('--color-surface-card')
  })

  it('shows processing ahead of the manual-annotation marker and uses constant-time indexes', () => {
    const processing = createImage('page-1', '001.png')
    processing.translationStatus = 'processing'
    processing.isManuallyAnnotated = true
    const wrapper = mountSidebar([processing])

    expect(wrapper.getComponent(ProductThumbnailGrid).props('items')).toMatchObject([{
      id: 0,
      cornerLabel: '处理中',
      disabledTitle: '正在处理',
    }])
    const source = readFileSync(
      resolve(process.cwd(), 'src/composables/useThumbnailSelection.ts'),
      'utf8',
    )
    expect(source).toContain('const imageIndexes = computed')
    expect(source).not.toContain('.findIndex(')
    expect(source).not.toContain('点击可重试')
  })

  it('keeps the sidebar panel owned by the thumbnail sidebar instead of an orphaned global class', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ThumbnailSidebar.vue'), 'utf8')

    expect(source).not.toContain('ui-panel-card')
    expect(source).toContain('class="thumbnail-sidebar__card"')
    expect(source).toContain('class="thumbnail-sidebar__back-button"')
    expect(source).toContain('class="thumbnail-sidebar__back-icon"')
    expect(source).toContain('class="thumbnail-sidebar__folder-content-list"')
    expect(source).toContain('class="thumbnail-sidebar__list"')
    expect(source).not.toContain('class="thumbnail-card"')
    expect(source).not.toContain('class="folder-back-btn"')
    expect(source).not.toContain('class="back-icon"')
    expect(source).not.toContain('class="folder-content-list"')
    expect(source).not.toContain('class="thumbnail-list"')
  })

  it('does not keep legacy DOM id hooks for thumbnail scrolling', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ThumbnailSidebar.vue'), 'utf8')

    expect(source).not.toContain('id="thumbnail-sidebar"')
    expect(source).not.toContain('id="thumbnailList"')
  })

  it('uses the shared button primitive contract for folder navigation actions', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ThumbnailSidebar.vue'), 'utf8')

    expect(source).toContain('variant="secondary"')
    expect(source).toContain('tone="primary"')
    expect(source).toContain('block')
    expect(source).not.toContain('--thumbnail-sidebar-folder-back-background')
    expect(source).not.toContain('linear-gradient(135deg, var(--thumbnail-sidebar-folder-back')
  })

  it('uses product status feedback for empty thumbnail states', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ThumbnailSidebar.vue'), 'utf8')

    expect(source).toContain('ProductStatusBanner')
    expect(source).not.toContain('class="empty-state"')
    expect(source).not.toContain('class="empty-folder"')

    const wrapper = mountSidebar([])
    const banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props()).toMatchObject({
      iconName: 'image',
      title: '暂无图片',
      tone: 'neutral',
    })
  })
})
