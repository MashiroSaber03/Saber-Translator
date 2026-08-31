import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it, vi } from 'vitest'

import CharacterDetailPanel from '@/components/insight/continuation/CharacterDetailPanel.vue'
import FormTile from '@/components/insight/continuation/FormTile.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductDetailPanel from '@/components/product/ProductDetailPanel.vue'
import ProductFileDropzone from '@/components/product/ProductFileDropzone.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiSwitch from '@/components/ui/UiSwitch.vue'

const characterDetailSourcePath = resolve(process.cwd(), 'src/components/insight/continuation/CharacterDetailPanel.vue')
const formTileSourcePath = resolve(process.cwd(), 'src/components/insight/continuation/FormTile.vue')

describe('continuation character controls', () => {
  it('exposes explicit names for character detail actions', () => {
    const wrapper = mount(CharacterDetailPanel, {
      props: {
        character: {
          name: 'Saber',
          aliases: [],
          description: '骑士王',
          forms: [],
          reference_image: '',
          enabled: true,
        },
        avatarUrl: '',
        getFormImageUrl: vi.fn(),
      },
      global: {
        stubs: {
          FormTile: true,
        },
      },
    })

    const toggle = wrapper.get('button[aria-label="启用角色 Saber"]')
    expect(toggle.attributes('aria-pressed')).toBeUndefined()
    expect(toggle.attributes('aria-checked')).toBe('true')
    expect(wrapper.find('button[aria-label="编辑角色 Saber"]').exists()).toBe(true)
    expect(wrapper.find('button[aria-label="删除角色 Saber"]').exists()).toBe(true)
    expect(wrapper.getComponent(ProductDetailPanel).props('ariaLabel')).toBe('角色 Saber 详情')
    expect(wrapper.getComponent(ProductActionRow).props('ariaLabel')).toBe('Saber 角色操作')

    const switchControl = wrapper.getComponent(UiSwitch)
    expect(switchControl.props('modelValue')).toBe(true)

    switchControl.vm.$emit('change', false)

    expect(wrapper.emitted('toggle-character')).toEqual([[false]])
  })

  it('renders detail and form empty states through product status feedback', () => {
    const emptyDetail = mount(CharacterDetailPanel, {
      props: {
        character: null,
        avatarUrl: '',
        getFormImageUrl: vi.fn(),
      },
      global: {
        stubs: {
          FormTile: true,
        },
      },
    })

    const detailBanner = emptyDetail.getComponent(ProductStatusBanner)
    expect(detailBanner.props('tone')).toBe('neutral')
    expect(detailBanner.props('role')).toBe('note')
    expect(detailBanner.props('iconName')).toBe('users')
    expect(emptyDetail.find('.empty-detail').exists()).toBe(false)

    const emptyForms = mount(CharacterDetailPanel, {
      props: {
        character: {
          name: 'Saber',
          aliases: [],
          description: '骑士王',
          forms: [],
          reference_image: '',
          enabled: true,
        },
        avatarUrl: '',
        getFormImageUrl: vi.fn(),
      },
      global: {
        stubs: {
          FormTile: true,
        },
      },
    })

    const formBanner = emptyForms.getComponent(ProductStatusBanner)
    expect(formBanner.props('iconName')).toBe('list')
    expect(formBanner.text()).toContain('点击“新增形态”添加')
    expect(emptyForms.find('.empty-forms').exists()).toBe(false)
  })

  it('uses the product avatar contract for character detail headers', () => {
    const source = readFileSync(characterDetailSourcePath, 'utf8')

    expect(source).toContain("import ProductAvatar from '@/components/product/ProductAvatar.vue'")
    expect(source).toContain('<ProductAvatar')
    expect(source).toContain('shape="rounded"')
    expect(source).not.toContain('class="detail-avatar"')
    expect(source).not.toContain('detail-avatar-placeholder')
    expect(source).not.toContain('character.name.charAt(0)')
  })

  it('does not override shared button primitive variables at the detail panel root', () => {
    const source = readFileSync(characterDetailSourcePath, 'utf8')
    const rootStyle = source.match(/\.character-detail-panel \{(?<body>[\s\S]*?)\n\}/)

    expect(rootStyle?.groups?.body ?? '').not.toMatch(/--ui-button-/)
  })

  it('uses the product section header contract for the form list heading', () => {
    const wrapper = mount(CharacterDetailPanel, {
      props: {
        character: {
          name: 'Saber',
          aliases: [],
          description: '骑士王',
          forms: [],
          reference_image: '',
          enabled: true,
        },
        avatarUrl: '',
        getFormImageUrl: vi.fn(),
      },
      global: {
        stubs: {
          FormTile: true,
        },
      },
    })
    const source = readFileSync(characterDetailSourcePath, 'utf8')
    const header = wrapper.getComponent(ProductSectionHeader)

    expect(header.props()).toMatchObject({ title: '形态列表' })
    expect(header.text()).toContain('新增形态')
    expect(source).toContain("import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'")
    expect(source).not.toContain('class="section-header"')
    expect(source).not.toContain('.section-header {')
    expect(source).not.toContain('.forms-section h4')
  })

  it('keeps character detail panel hooks under the panel owner', () => {
    const source = readFileSync(characterDetailSourcePath, 'utf8')

    for (const oldClass of [
      'empty-detail-status',
      'detail-header',
      'detail-main-info',
      'detail-info',
      'detail-aliases',
      'detail-actions',
      'forms-section',
      'empty-forms-status',
      'forms-grid',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldClass}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldClass}\\b`))
    }
    expect(source).not.toContain("'has-selection': !!character")
    expect(source).not.toMatch(/\.character-detail-panel__[^{]+ h4/)

    for (const ownerClass of [
      'character-detail-panel--has-selection',
      'character-detail-panel__empty-status',
      'character-detail-panel__header',
      'character-detail-panel__main-info',
      'character-detail-panel__info',
      'character-detail-panel__title',
      'character-detail-panel__aliases',
      'character-detail-panel__actions',
      'character-detail-panel__forms-section',
      'character-detail-panel__empty-forms-status',
      'character-detail-panel__forms-grid',
    ]) {
      expect(source).toContain(ownerClass)
    }
  })

  it('uses icon-button primitives for character detail icon-only actions', () => {
    const wrapper = mount(CharacterDetailPanel, {
      props: {
        character: {
          name: 'Saber',
          aliases: [],
          description: '骑士王',
          forms: [],
          reference_image: '',
          enabled: true,
        },
        avatarUrl: '',
        getFormImageUrl: vi.fn(),
      },
      global: {
        stubs: {
          FormTile: true,
        },
      },
    })
    const source = readFileSync(characterDetailSourcePath, 'utf8')

    expect(source).toContain("import UiIconButton from '@/components/ui/UiIconButton.vue'")
    expect(wrapper.findAllComponents(UiIconButton).map(button => button.props('label'))).toEqual([
      '编辑角色 Saber',
      '删除角色 Saber',
    ])
    expect(source).not.toContain('<UiButton variant="toolbar" :aria-label="`编辑角色')
    expect(source).not.toContain('<UiButton variant="danger" :aria-label="`删除角色')
  })

  it('exposes explicit names for form-tile controls', () => {
    const wrapper = mount(FormTile, {
      props: {
        form: {
          form_id: 'form_1',
          form_name: '常服',
          description: '日常服装',
          reference_image: '/tmp/form.png',
          enabled: true,
        },
        characterName: 'Saber',
        formImageUrl: '/tmp/form.png',
      },
    })

    const dropzone = wrapper.getComponent(ProductFileDropzone)
    expect(dropzone.props()).toMatchObject({
      inputId: 'formTileUpload-form_1',
      accept: '.png,.jpg,.jpeg,.webp,.gif,.bmp,.tif,.tiff',
      label: '上传 Saber 常服 参考图',
    })
    expect(readFileSync(formTileSourcePath, 'utf8')).not.toContain("import UiFileInput from '@/components/ui/UiFileInput.vue'")
    const toggle = wrapper.get('button[aria-label="启用 Saber 常服"]')
    expect(toggle.attributes('aria-pressed')).toBeUndefined()
    expect(toggle.attributes('aria-checked')).toBe('true')
    expect(wrapper.find('button[aria-label="生成 Saber 常服 三视图"]').exists()).toBe(true)
    expect(wrapper.find('button[aria-label="删除 Saber 常服 参考图"]').exists()).toBe(true)
    expect(wrapper.find('button[aria-label="编辑 Saber 常服"]').exists()).toBe(true)
    expect(wrapper.find('button[aria-label="删除 Saber 常服"]').exists()).toBe(true)
    expect(wrapper.get('img.form-tile__image').attributes('alt')).toBe('Saber 常服参考图')

    const switchControl = wrapper.getComponent(UiSwitch)
    expect(switchControl.props('size')).toBe('sm')

    switchControl.vm.$emit('change', false)

    expect(wrapper.emitted('toggle-enabled')).toEqual([[false]])
  })

  it('uses product card, action rows, and chip contracts for form tiles', () => {
    const wrapper = mount(FormTile, {
      props: {
        form: {
          form_id: 'form_1',
          form_name: '常服',
          description: '日常服装',
          reference_image: '',
          enabled: false,
        },
        characterName: 'Saber',
        formImageUrl: '',
      },
    })

    expect(wrapper.getComponent(ProductRecordCard).exists()).toBe(true)

    const actionRows = wrapper.findAllComponents(ProductActionRow)
    expect(actionRows.map(row => row.props('ariaLabel'))).toEqual([
      'Saber 常服三视图操作',
      'Saber 常服形态管理操作',
    ])

    const statusChips = wrapper.getComponent(ProductChipList)
    expect(statusChips.props('items')).toEqual([
      { id: 'disabled', label: '已禁用', tone: 'warning' },
    ])
  })

  it('keeps form-tile internals on owner-scoped class names', () => {
    const source = readFileSync(formTileSourcePath, 'utf8')

    for (const oldClass of [
      'form-image-section',
      'form-image-empty-state',
      'upload-overlay',
      'upload-text',
      'form-content',
      'form-header',
      'form-title',
      'form-description',
      'form-actions',
      'action-row',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldClass}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldClass}\\b`))
    }
    expect(source).not.toContain('class="action-row secondary"')
    expect(source).not.toContain('.action-row.secondary')

    for (const ownerClass of [
      'form-tile__image-section',
      'form-tile__upload-overlay',
      'form-tile__upload-text',
      'form-tile__content',
      'form-tile__header',
      'form-tile__actions',
      'form-tile__action-row',
      'form-tile__action-row--secondary',
    ]) {
      expect(source).toContain(ownerClass)
    }
  })

  it('uses the compact product empty state for missing form reference images', () => {
    const source = readFileSync(formTileSourcePath, 'utf8')
    expect(source).toContain("import ProductEmptyState from '@/components/product/ProductEmptyState.vue'")
    expect(source).not.toContain('placeholder-text')
    expect(source).not.toContain('placeholder-icon')
    expect(source).not.toContain('image-placeholder')

    const wrapper = mount(FormTile, {
      props: {
        form: {
          form_id: 'form_1',
          form_name: '常服',
          description: '日常服装',
          reference_image: '',
          enabled: true,
        },
        characterName: 'Saber',
        formImageUrl: '',
      },
    })

    const emptyState = wrapper.getComponent(ProductEmptyState)
    expect(emptyState.props()).toMatchObject({
      iconName: 'camera',
      role: 'note',
      size: 'compact',
      title: '未上传参考图',
    })
  })

  it('uses icon-button primitives for form tile icon-only actions', () => {
    const wrapper = mount(FormTile, {
      props: {
        form: {
          form_id: 'form_1',
          form_name: '常服',
          description: '日常服装',
          reference_image: '/tmp/form.png',
          enabled: true,
        },
        characterName: 'Saber',
        formImageUrl: '/tmp/form.png',
      },
    })
    const source = readFileSync(formTileSourcePath, 'utf8')

    expect(source).toContain("import UiIconButton from '@/components/ui/UiIconButton.vue'")
    expect(wrapper.findAllComponents(UiIconButton).map(button => button.props('label'))).toEqual([
      '生成 Saber 常服 三视图',
      '删除 Saber 常服 参考图',
      '编辑 Saber 常服',
      '删除 Saber 常服',
    ])
    expect(source).not.toContain('<UiButton variant="secondary" class="form-tile__primary-action"')
    expect(source).not.toContain('<UiButton v-if="form.reference_image" variant="danger"')
    expect(source).not.toContain('<UiButton variant="toolbar" :aria-label="`编辑')
  })

  it('reveals the form tile upload overlay for keyboard focus as well as hover', () => {
    const source = readFileSync(formTileSourcePath, 'utf8')

    expect(source).toContain('.form-tile__image-section:focus-within .form-tile__upload-overlay')
  })

  it('maps form tile owner colors through semantic tokens', () => {
    const source = readFileSync(formTileSourcePath, 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    expect(styleBlock).toContain('--shadow-soft')
  })
})
