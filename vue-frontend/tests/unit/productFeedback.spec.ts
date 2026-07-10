import { mount } from '@vue/test-utils'
import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import ProductAvatar from '@/components/product/ProductAvatar.vue'
import ProductLogPanel from '@/components/product/ProductLogPanel.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductComposer from '@/components/product/ProductComposer.vue'
import ProductDetailPanel from '@/components/product/ProductDetailPanel.vue'
import ProductDetailSection from '@/components/product/ProductDetailSection.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import ProductFolderCard from '@/components/product/ProductFolderCard.vue'
import ProductMessageBubble from '@/components/product/ProductMessageBubble.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductBookSelector from '@/components/product/ProductBookSelector.vue'
import ProductBreadcrumbTrail from '@/components/product/ProductBreadcrumbTrail.vue'
import ProductSearchField from '@/components/product/ProductSearchField.vue'
import ProductChoiceCardGrid from '@/components/product/ProductChoiceCardGrid.vue'
import ProductCardGrid from '@/components/product/ProductCardGrid.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductThumbnailGrid from '@/components/product/ProductThumbnailGrid.vue'
import ProductSelectableImageGrid from '@/components/product/ProductSelectableImageGrid.vue'
import ProductScrollStack from '@/components/product/ProductScrollStack.vue'
import ProductHorizontalScrollStrip from '@/components/product/ProductHorizontalScrollStrip.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiCombobox from '@/components/ui/UiCombobox.vue'
import UiIcon from '@/components/ui/UiIcon.vue'

describe('product feedback components', () => {
  it('owns reusable avatar image and initial fallback surfaces', () => {
    const imageWrapper = mount(ProductAvatar, {
      props: {
        imageSrc: '/avatar.png',
        label: '角色头像 Saber',
        fallbackText: 'S',
        size: 'md',
        shape: 'rounded',
      },
    })

    const image = imageWrapper.get('img')
    expect(image.attributes('src')).toBe('/avatar.png')
    expect(image.attributes('alt')).toBe('角色头像 Saber')
    expect(imageWrapper.get('.product-avatar').classes()).toContain('product-avatar--md')
    expect(imageWrapper.get('.product-avatar').classes()).toContain('product-avatar--rounded')

    const fallbackWrapper = mount(ProductAvatar, {
      props: {
        label: '角色头像 未命名',
        fallbackText: '',
        size: 'hero',
        shape: 'portrait',
      },
    })

    expect(fallbackWrapper.get('.product-avatar').attributes('aria-label')).toBe('角色头像 未命名')
    expect(fallbackWrapper.get('.product-avatar').classes()).toContain('product-avatar--hero')
    expect(fallbackWrapper.get('.product-avatar').classes()).toContain('product-avatar--portrait')
    expect(fallbackWrapper.get('.product-avatar__fallback').text()).toBe('角')
  })

  it('renders danger status banners with alert semantics', () => {
    const wrapper = mount(ProductStatusBanner, {
      props: { tone: 'danger' },
      slots: { default: '操作失败' },
    })

    expect(wrapper.attributes('role')).toBe('alert')
    expect(wrapper.text()).toContain('操作失败')
  })

  it('renders empty states with typed icons and action content', () => {
    const wrapper = mount(ProductEmptyState, {
      props: {
        eyebrow: '缺少上下文',
        iconName: 'book-open',
        title: '书架空空如也',
        description: '点击新建书籍开始',
      },
      slots: {
        actions: '<button>新建</button>',
      },
    })

    expect(wrapper.getComponent(UiIcon).props('name')).toBe('book-open')
    expect(wrapper.get('.product-empty-state__eyebrow').text()).toBe('缺少上下文')
    expect(wrapper.get('h2').text()).toBe('书架空空如也')
    expect(wrapper.text()).toContain('点击新建书籍开始')
    expect(wrapper.get('.product-empty-state__actions').text()).toContain('新建')
  })

  it('owns the reusable slot-based card grid layout', () => {
    const wrapper = mount(ProductCardGrid, {
      props: {
        ariaLabel: '书籍卡片列表',
        minItemWidth: '160px',
        gap: '24px',
      },
      slots: {
        default: '<button class="test-card">Book</button>',
      },
    })

    expect(wrapper.classes()).toContain('product-card-grid')
    expect(wrapper.attributes('role')).toBe('group')
    expect(wrapper.attributes('aria-label')).toBe('书籍卡片列表')
    expect(wrapper.attributes('style')).toContain('--product-card-grid-min-item-width: 160px;')
    expect(wrapper.attributes('style')).toContain('--product-card-grid-gap: 24px;')
    expect(wrapper.get('.test-card').text()).toBe('Book')
  })

  it('supports inverse empty-state presentation for dark work surfaces', () => {
    const wrapper = mount(ProductEmptyState, {
      props: {
        iconName: 'book-open',
        title: '暂无图片',
        variant: 'inverse',
      },
    })

    expect(wrapper.classes()).toContain('product-empty-state--inverse')
  })

  it('supports compact empty-state presentation for cards and media slots', () => {
    const wrapper = mount(ProductEmptyState, {
      props: {
        iconName: 'camera',
        title: '未上传参考图',
        size: 'compact',
        role: 'note',
      },
    })

    expect(wrapper.classes()).toContain('product-empty-state--compact')
    expect(wrapper.attributes('role')).toBe('note')
    expect(wrapper.getComponent(UiIcon).props('name')).toBe('camera')
    expect(wrapper.get('h2').text()).toBe('未上传参考图')
  })

  it('renders collapsible log rows through a product log panel', async () => {
    const wrapper = mount(ProductLogPanel, {
      props: {
        expanded: true,
        title: '运行日志',
        items: [
          { id: 1, timestamp: '12:00', message: '准备任务', tone: 'info' },
          { id: 2, timestamp: '12:01', message: '任务完成', tone: 'success' },
        ],
      },
    })

    expect(wrapper.get('[role="log"]').text()).toContain('任务完成')
    const logContentId = wrapper.get('[role="log"]').attributes('id')
    expect(logContentId).toMatch(/\S/)
    expect(wrapper.get('.product-log-panel__header').attributes('aria-controls')).toBe(logContentId)

    await wrapper.get('.product-log-panel__header').trigger('click')
    expect(wrapper.emitted('toggle')).toBeTruthy()
  })

  it('keeps product log panel font weights numeric', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/product/ProductLogPanel.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/font-weight:\s*(bold|normal)\b/)
  })

  it('supports structured log details for debug payloads', () => {
    const wrapper = mount(ProductLogPanel, {
      props: {
        expanded: true,
        title: '调试事件',
        items: [
          {
            id: 'event-1',
            message: 'debug_result',
            detail: '{\n  "group_id": "tool-1"\n}',
            tone: 'accent',
          },
        ],
      },
    })

    expect(wrapper.get('.product-log-panel__message').text()).toBe('debug_result')
    expect(wrapper.get('.product-log-panel__detail').text()).toContain('"group_id": "tool-1"')
  })

  it('owns reusable search-field input, submit, and clear semantics', async () => {
    const wrapper = mount(ProductSearchField, {
      props: {
        modelValue: 'Saber',
        ariaLabel: '搜索书籍',
        placeholder: '搜索书籍名称',
      },
    })

    expect(wrapper.getComponent(UiIcon).props('name')).toBe('search')
    expect(wrapper.get('input').attributes('type')).toBe('search')
    expect(wrapper.get('input').attributes('aria-label')).toBe('搜索书籍')

    await wrapper.get('input').setValue('Rin')
    expect(wrapper.emitted('update:modelValue')?.[0]).toEqual(['Rin'])

    await wrapper.setProps({ modelValue: 'Rin' })
    await wrapper.get('input').trigger('keydown.enter')
    expect(wrapper.emitted('search')?.[0]).toEqual(['Rin'])

    await wrapper.get('button[aria-label="清除搜索"]').trigger('click')
    expect(wrapper.emitted('update:modelValue')?.at(-1)).toEqual([''])
    expect(wrapper.emitted('clear')).toBeTruthy()

    const source = readFileSync(resolve(process.cwd(), 'src/components/product/ProductSearchField.vue'), 'utf8')
    expect(source).not.toContain('event.target as HTMLInputElement')
    expect(source).not.toContain('@input="handleInput"')
    expect(source).not.toContain('input: [event: Event]')
  })

  it('keeps selectable image cards on checkbox semantics', async () => {
    const wrapper = mount(ProductSelectableImageGrid, {
      props: {
        items: [
          {
            id: 1,
            src: 'https://example.com/1.jpg',
            alt: '第一页',
            label: '第 1 页',
            selected: false,
          },
        ],
      },
    })

    const checkbox = wrapper.getComponent(UiCheckbox)
    expect(checkbox.props('ariaLabel')).toBe('选择第 1 页')

    checkbox.vm.$emit('change', true)
    expect(wrapper.emitted('toggle')?.[0]).toEqual([1])
  })

  it('owns reusable exclusive choice-card selection semantics', async () => {
    const wrapper = mount(ProductChoiceCardGrid, {
      props: {
        ariaLabel: '导出格式',
        modelValue: 'images',
        items: [
          { id: 'images', label: '图片 ZIP', description: '所有页面打包下载', iconName: 'image' },
          { id: 'pdf', label: 'PDF 文档', description: '方便阅读和分享', iconName: 'file-text' },
        ],
      },
    })

    expect(wrapper.get('[role="radiogroup"]').attributes('aria-label')).toBe('导出格式')

    const options = wrapper.findAll('[role="radio"]')
    expect(options).toHaveLength(2)
    expect(options[0].attributes('aria-checked')).toBe('true')
    expect(options[1].attributes('aria-checked')).toBe('false')

    await options[1].trigger('click')

    expect(wrapper.emitted('update:modelValue')).toEqual([['pdf']])
    expect(wrapper.emitted('select')).toEqual([['pdf']])
  })

  it('renders clickable thumbnail cards with selected and marked states', async () => {
    const wrapper = mount(ProductThumbnailGrid, {
      props: {
        ariaLabel: '页面导航',
        items: [
          {
            id: 1,
            src: 'https://example.com/1.jpg',
            alt: '第一页缩略图',
            label: '第 1 页',
            selected: true,
            marked: true,
          },
        ],
      },
    })

    const button = wrapper.get('button[aria-label="选择第 1 页"]')
    expect(wrapper.get('[role="list"]').attributes('aria-label')).toBe('页面导航')
    expect(button.attributes('aria-pressed')).toBe('true')
    expect(button.attributes('data-product-thumbnail-id')).toBe('1')
    expect(button.classes()).toContain('product-thumbnail-grid__item--marked')

    await button.trigger('click')
    expect(wrapper.emitted('select')?.[0]).toEqual([1])

    await wrapper.get('img').trigger('error')
    expect(wrapper.text()).toContain('第 1 页')
  })

  it('supports thumbnail badges fallback copy and explicit action names', () => {
    const wrapper = mount(ProductThumbnailGrid, {
      props: {
        ariaLabel: '参考图选择',
        items: [
          {
            id: 'original:1',
            src: '',
            alt: '第1页',
            label: '1',
            selected: true,
            selectedBadge: '1',
            cornerLabel: '续写',
            fallbackLabel: '占位页',
            ariaLabel: '取消选择原作第1页参考图',
            disabled: true,
            disabledTitle: '已达到最大数量，请先取消其他选择',
          },
        ],
      },
    })

    const button = wrapper.get('button[aria-label="取消选择原作第1页参考图"]')
    expect(button.attributes('title')).toBe('已达到最大数量，请先取消其他选择')
    expect(wrapper.text()).toContain('占位页')
    expect(wrapper.text()).toContain('续写')
    expect(wrapper.text()).toContain('1')
  })

  it('renders static thumbnail cards without exposing button behavior', () => {
    const wrapper = mount(ProductThumbnailGrid, {
      props: {
        ariaLabel: '角色档案',
        items: [
          {
            id: 'character-form-1',
            src: 'https://example.com/form.jpg',
            alt: 'Saber - 常服',
            label: 'Saber - 常服',
            interactive: false,
          },
        ],
      },
    })

    expect(wrapper.find('button[aria-label="选择Saber - 常服"]').exists()).toBe(false)
    expect(wrapper.get('[role="img"]').attributes('aria-label')).toBe('Saber - 常服')
    expect(wrapper.emitted('select')).toBeUndefined()
  })

  it('uses explicit product image hooks inside reusable image grids', () => {
    const thumbnailSource = readFileSync(resolve(process.cwd(), 'src/components/product/ProductThumbnailGrid.vue'), 'utf8')
    const selectableSource = readFileSync(resolve(process.cwd(), 'src/components/product/ProductSelectableImageGrid.vue'), 'utf8')

    expect(thumbnailSource).toContain('product-thumbnail-grid__preview-image')
    expect(thumbnailSource).not.toContain('.product-thumbnail-grid__preview img')
    expect(selectableSource).toContain('product-selectable-image-grid__preview-image')
    expect(selectableSource).not.toContain('.product-selectable-image-grid__preview img')
  })

  it('owns reusable breadcrumb trail navigation semantics', async () => {
    const wrapper = mount(ProductBreadcrumbTrail, {
      props: {
        items: [
          { path: '', name: '根目录' },
          { path: 'chapter-a', name: 'chapter-a' },
        ],
      },
    })

    const nav = wrapper.get('nav[aria-label="当前位置"]')
    expect(nav.text()).toContain('根目录')
    expect(nav.text()).toContain('chapter-a')
    expect(wrapper.get('[aria-current="page"]').text()).toContain('chapter-a')

    await wrapper.get('button[aria-label="打开根目录"]').trigger('click')
    expect(wrapper.emitted('select')?.[0]).toEqual([''])
  })

  it('owns reusable scroll-stack semantics and bottom scrolling', () => {
    const wrapper = mount(ProductScrollStack, {
      props: {
        role: 'log',
        ariaLabel: '活动记录',
        ariaLive: 'polite',
        empty: true,
      },
      slots: {
        empty: '<p>暂无记录</p>',
      },
    })

    const scroller = wrapper.get('[role="log"]')
    Object.defineProperty(scroller.element, 'scrollHeight', { configurable: true, value: 320 })

    expect(scroller.attributes('aria-label')).toBe('活动记录')
    expect(scroller.attributes('aria-live')).toBe('polite')
    expect(wrapper.text()).toContain('暂无记录')

    wrapper.vm.scrollToBottom()
    expect((scroller.element as HTMLElement).scrollTop).toBe(320)
  })

  it('owns reusable horizontal scroll-strip semantics', () => {
    const wrapper = mount(ProductHorizontalScrollStrip, {
      props: {
        ariaLabel: '横向素材列表',
      },
      slots: {
        default: '<button>第一页</button><button>第二页</button>',
      },
    })

    const strip = wrapper.get('[role="region"]')
    expect(strip.attributes('aria-label')).toBe('横向素材列表')
    expect(strip.classes()).toContain('product-horizontal-scroll-strip')
    expect(strip.text()).toContain('第一页')
    expect(strip.text()).toContain('第二页')
  })

  it('renders interactive and static product chips with list semantics', async () => {
    const wrapper = mount(ProductChipList, {
      props: {
        ariaLabel: '引用和标签',
        items: [
          { id: 5, label: '第5页', ariaLabel: '查看第 5 页', interactive: true, tone: 'primary' },
          { id: 'tag-story', label: '剧情', tone: 'neutral' },
          { id: 'failed', label: '失败', tone: 'danger' },
        ],
      },
    })

    expect(wrapper.get('[role="list"]').attributes('aria-label')).toBe('引用和标签')
    expect(wrapper.text()).toContain('剧情')
    expect(wrapper.find('.product-chip-list__chip--danger').exists()).toBe(true)

    const citationButton = wrapper.get('button[aria-label="查看第 5 页"]')
    expect(citationButton.attributes('aria-pressed')).toBeUndefined()

    await citationButton.trigger('click')

    expect(wrapper.emitted('select')?.[0]).toEqual([5])
  })

  it('owns reusable folder-card navigation metadata', async () => {
    const componentPath = resolve(process.cwd(), 'src/components/product/ProductFolderCard.vue')

    expect(existsSync(componentPath)).toBe(true)

    const source = readFileSync(componentPath, 'utf8')
    expect(source).toContain('ProductRecordCard')
    expect(source).toContain('ProductChipList')
    expect(source).toContain('folderName')
    expect(source).toContain('countId?: string | number')

    const wrapper = mount(ProductFolderCard, {
      props: {
        count: 2,
        folderName: 'chapter-a',
      },
    })

    const folderCard = wrapper.getComponent(ProductRecordCard)
    expect(folderCard.props('as')).toBe('button')
    expect(folderCard.props('ariaLabel')).toBe('打开文件夹 chapter-a')
    expect(wrapper.getComponent(ProductChipList).props('items')).toEqual([
      {
        id: 'chapter-a-count',
        label: '2 张',
        tone: 'neutral',
      },
    ])

    const pathScopedWrapper = mount(ProductFolderCard, {
      props: {
        count: 2,
        countId: 'root/chapter-a',
        folderName: 'chapter-a',
      },
    })
    expect(pathScopedWrapper.getComponent(ProductChipList).props('items')[0]).toMatchObject({
      id: 'root/chapter-a',
      label: '2 张',
    })

    await folderCard.trigger('click')
    expect(wrapper.emitted('select')).toEqual([[]])
  })

  it('exposes selected interactive chips as pressed product controls', () => {
    const wrapper = mount(ProductChipList, {
      props: {
        ariaLabel: '标签筛选',
        items: [
          {
            id: 'Drama',
            label: 'Drama',
            ariaLabel: 'Drama',
            interactive: true,
            selected: true,
            tone: 'custom',
            backgroundColor: '#aa6644',
          },
        ],
      },
    })

    const selectedChip = wrapper.get('button[aria-label="Drama"]')

    expect(selectedChip.attributes('aria-pressed')).toBe('true')
    expect(selectedChip.classes()).toContain('product-chip-list__chip--selected')
  })

  it('renders user-colored product chips through owner CSS variables', () => {
    const wrapper = mount(ProductChipList, {
      props: {
        ariaLabel: '书籍标签',
        items: [
          {
            id: 'Drama',
            label: 'Drama',
            tone: 'custom',
            backgroundColor: '#aa6644',
            textColor: 'var(--color-text-inverse)',
          },
        ],
      },
    })

    const customChip = wrapper.get('.product-chip-list__chip--custom')
    expect(customChip.attributes('style')).toContain('--product-chip-list-custom-background: #aa6644;')
    expect(customChip.attributes('style')).toContain('--product-chip-list-custom-text: var(--color-text-inverse);')
  })

  it('maps inverse chip chrome through semantic overlay tokens', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/product/ProductChipList.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(|rgb\(/)
    expect(styleBlock).toContain('--color-overlay-inverse-emphasis')
    expect(styleBlock).toContain('--color-overlay-inverse-muted')
  })

  it('owns text composer submit semantics', async () => {
    const wrapper = mount(ProductComposer, {
      props: {
        modelValue: '这个章节讲了什么？',
        placeholder: '输入问题',
        submitLabel: '发送',
      },
    })

    expect(wrapper.get('textarea').attributes('aria-label')).toBe('输入问题')

    await wrapper.get('textarea').trigger('keydown', { key: 'Enter' })
    expect(wrapper.emitted('submit')).toBeTruthy()

    await wrapper.get('textarea').trigger('keydown', { key: 'Enter', shiftKey: true })
    expect(wrapper.emitted('submit')).toHaveLength(1)

    await wrapper.get('button[aria-label="发送"]').trigger('click')
    expect(wrapper.emitted('submit')).toHaveLength(2)
  })

  it('owns message bubble role, avatar, and actions slots', () => {
    const wrapper = mount(ProductMessageBubble, {
      props: {
        role: 'assistant',
        appearance: 'reading',
        avatarLabel: '智能助手',
        avatarIconName: 'message',
        ariaLabel: '助手回复',
      },
      slots: {
        default: '<p>回答内容</p>',
        actions: '<button type="button">保存</button>',
      },
    })

    expect(wrapper.get('article').attributes('aria-label')).toBe('助手回复')
    expect(wrapper.get('article').classes()).toContain('product-message-bubble--appearance-reading')
    expect(wrapper.get('.product-message-bubble__avatar').attributes('aria-label')).toBe('智能助手')
    expect(wrapper.get('.product-message-bubble__body').text()).toContain('回答内容')
    expect(wrapper.get('.product-message-bubble__actions').text()).toContain('保存')

    const source = readFileSync(resolve(process.cwd(), 'src/components/product/ProductMessageBubble.vue'), 'utf8')
    expect(source).toMatch(/\.product-message-bubble__content\s*\{[\s\S]*overflow-wrap:\s*anywhere/)
  })

  it('owns reusable record-card slots and accent state', () => {
    const wrapper = mount(ProductRecordCard, {
      props: {
        accent: true,
        ariaLabel: '笔记：角色动机',
      },
      attrs: {
        role: 'listitem',
      },
      slots: {
        icon: '<span data-test="icon">图标</span>',
        meta: '<span>刚刚</span>',
        actions: '<button type="button">编辑</button>',
        default: '<p>正文</p>',
        footer: '<span>引用</span>',
      },
    })

    const card = wrapper.get('.product-record-card')
    expect(card.attributes('aria-label')).toBe('笔记：角色动机')
    expect(card.attributes('role')).toBe('listitem')
    expect(card.classes()).toContain('product-record-card--accent')
    expect(wrapper.get('.product-record-card__meta').text()).toContain('刚刚')
    expect(wrapper.get('.product-record-card__actions').text()).toContain('编辑')
    expect(wrapper.get('.product-record-card__body').text()).toContain('正文')
    expect(wrapper.get('.product-record-card__footer').text()).toContain('引用')
  })

  it('owns reusable detail panel and section surfaces', () => {
    const wrapper = mount(ProductDetailPanel, {
      props: {
        ariaLabel: '问答预览',
      },
      slots: {
        default: `
          <ProductDetailSection label="问题">
            <template #label-actions><button type="button">展开</button></template>
            发生了什么？
          </ProductDetailSection>
          <ProductDetailSection label="回答" scroll>回答内容</ProductDetailSection>
          <ProductDetailSection label="引用页码" :framed="false"><button type="button">第7页</button></ProductDetailSection>
        `,
      },
      global: {
        components: {
          ProductDetailSection,
        },
      },
    })

    expect(wrapper.get('.product-detail-panel').attributes('aria-label')).toBe('问答预览')
    const sections = wrapper.findAllComponents(ProductDetailSection)
    expect(sections.map(section => section.props('label'))).toEqual(['问题', '回答', '引用页码'])
    expect(wrapper.get('.product-detail-section__label-actions').text()).toContain('展开')
    expect(wrapper.get('.product-detail-section__content--scroll').text()).toContain('回答内容')
    expect(wrapper.findAll('.product-detail-section__content--framed')).toHaveLength(2)
  })

  it('owns reusable section headers with optional icon, description, and actions', () => {
    const wrapper = mount(ProductSectionHeader, {
      props: {
        title: '角色档案',
        description: '点击角色查看和管理形态',
        iconName: 'users',
      },
      slots: {
        actions: '<button type="button">新增角色</button>',
      },
    })

    expect(wrapper.get('.product-section-header').exists()).toBe(true)
    expect(wrapper.getComponent(UiIcon).props('name')).toBe('users')
    expect(wrapper.get('.product-section-header__title').text()).toBe('角色档案')
    expect(wrapper.get('.product-section-header__description').text()).toBe('点击角色查看和管理形态')
    expect(wrapper.get('.product-section-header__actions').text()).toContain('新增角色')
  })

  it('supports compact section headers for dense side panels and toolbars', () => {
    const wrapper = mount(ProductSectionHeader, {
      props: {
        title: '内容导航',
        size: 'sm',
      },
    })

    expect(wrapper.get('.product-section-header').classes()).toContain('product-section-header--sm')
    expect(wrapper.find('.product-section-header__description').exists()).toBe(false)
    expect(wrapper.find('.product-section-header__actions').exists()).toBe(false)
  })

  it('keeps section headers responsive when long titles share space with actions', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/product/ProductSectionHeader.vue'), 'utf8')

    expect(source).toMatch(/\.product-section-header__copy\s*\{[\s\S]*flex:\s*1 1 auto/)
    expect(source).toMatch(/\.product-section-header__title\s*\{[\s\S]*min-width:\s*0/)
    expect(source).toMatch(/\.product-section-header__title\s*\{[\s\S]*overflow-wrap:\s*anywhere/)
    expect(source).toContain('product-section-header__title-text')
    expect(source).toMatch(/\.product-section-header__title-text\s*\{[\s\S]*min-width:\s*0/)
    expect(source).toMatch(/\.product-section-header__title-text\s*\{[\s\S]*overflow-wrap:\s*anywhere/)
    expect(source).not.toContain('.product-section-header__title span')
  })

  it('owns reusable book selector options and select events', () => {
    const wrapper = mount(ProductBookSelector, {
      props: {
        modelValue: '',
        books: [
          { id: 'book-1', title: '第一本书' },
          { id: 'book-2', title: '' },
        ],
      },
    })

    const combobox = wrapper.getComponent(UiCombobox)
    expect(combobox.props('fit')).toBe(true)
    expect(combobox.props('options')).toEqual([
      { label: '选择书籍', value: '' },
      { label: '第一本书', value: 'book-1' },
      { label: 'book-2', value: 'book-2' },
    ])

    combobox.vm.$emit('change', 'book-1')

    expect(wrapper.emitted('update:modelValue')).toEqual([['book-1']])
    expect(wrapper.emitted('select')).toEqual([['book-1']])
  })

  it('owns reusable action-row layout for panel footers', () => {
    const wrapper = mount(ProductActionRow, {
      props: {
        ariaLabel: '续写步骤操作',
        divider: true,
        justify: 'between',
        variant: 'dialog',
      },
      slots: {
        default: '<button type="button">上一步</button><button type="button">下一步</button>',
      },
    })

    expect(wrapper.attributes('role')).toBe('group')
    expect(wrapper.attributes('aria-label')).toBe('续写步骤操作')
    expect(wrapper.classes()).toContain('product-action-row--between')
    expect(wrapper.classes()).toContain('product-action-row--divider')
    expect(wrapper.classes()).toContain('product-action-row--dialog')
    expect(wrapper.text()).toContain('下一步')
  })

  it('owns reusable product form-section surfaces outside the UI primitive layer', () => {
    const wrapper = mount(ProductFormSection, {
      slots: {
        title: '<h3>模型设置</h3>',
        default: '<label>Provider</label>',
      },
    })

    expect(wrapper.get('section').classes()).toEqual(expect.arrayContaining([
      'product-form-section',
      'product-form-section--padded',
    ]))
    expect(wrapper.get('.product-form-section__title').text()).toContain('模型设置')
    expect(wrapper.text()).toContain('Provider')

    const formSectionSource = readFileSync(resolve(process.cwd(), 'src/components/product/ProductFormSection.vue'), 'utf8')

    expect(formSectionSource).toContain('product-form-section')
    expect(formSectionSource).toContain(':slotted(.ui-form-hint)')
    expect(formSectionSource).not.toContain('ui-panel--settings')
  })

  it('supports centered action groups for toolbars and result actions', () => {
    const wrapper = mount(ProductActionRow, {
      props: {
        ariaLabel: '翻译结果操作',
        justify: 'center',
        variant: 'toolbar',
      },
      slots: {
        default: '<button type="button">下载</button>',
      },
    })

    expect(wrapper.attributes('role')).toBe('group')
    expect(wrapper.attributes('aria-label')).toBe('翻译结果操作')
    expect(wrapper.classes()).toContain('product-action-row--center')
    expect(wrapper.classes()).toContain('product-action-row--toolbar')

    const source = readFileSync(resolve(process.cwd(), 'src/components/product/ProductActionRow.vue'), 'utf8')
    expect(source).toContain("'toolbar'")
    expect(source).toContain('.product-action-row--toolbar')
    expect(source).toContain('--ui-button-sm-padding')
  })
})
