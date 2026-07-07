import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineComponent, h, ref } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { confirmProductActionMock } = vi.hoisted(() => ({
  confirmProductActionMock: vi.fn(),
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

const characterDetailPanelStub = defineComponent({
  props: {
    character: {
      type: Object,
      default: null,
    },
  },
  setup(props, { emit }) {
    return () => h('div', { class: 'character-detail-stub' }, [
      h('span', { class: 'detail-name' }, props.character?.name || 'empty'),
      h('button', {
        type: 'button',
        class: 'delete-form',
        onClick: () => emit('delete-form', { form_id: 'form-1', form_name: '常服' }),
      }, '删除形态'),
      h('button', {
        type: 'button',
        class: 'delete-form-image',
        onClick: () => emit('delete-form-image', 'form-1'),
      }, '删除形态参考图'),
      h('button', {
        type: 'button',
        class: 'delete-character',
        onClick: () => emit('delete-character'),
      }, '删除角色'),
    ])
  },
})

import CharacterManagementPanel from '@/components/insight/continuation/CharacterManagementPanel.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'

const componentSourcePath = resolve(process.cwd(), 'src/components/insight/continuation/CharacterManagementPanel.vue')

function createState() {
  return {
    characters: ref([
      {
        name: '主角',
        aliases: [],
        description: 'desc',
        forms: [{ form_id: 'form-1', form_name: '常服', description: '', reference_image: '/tmp/form.png' }],
        reference_image: '',
        enabled: true,
      },
    ]),
    getCharacterImageUrl: vi.fn().mockReturnValue(''),
    getFormImageUrl: vi.fn().mockReturnValue(''),
    showMessage: vi.fn(),
  }
}

describe('CharacterManagementPanel', () => {
  beforeEach(() => {
    confirmProductActionMock.mockReset()
    confirmProductActionMock.mockResolvedValue(true)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('uses button semantics for selectable character tiles', async () => {
    const wrapper = mount(CharacterManagementPanel, {
      props: {
        bookId: 'book-1',
        characterManagement: {},
        state: createState(),
      },
      global: {
        stubs: {
          CharacterDetailPanel: characterDetailPanelStub,
          AddCharacterDialog: true,
          EditCharacterDialog: true,
          AddFormDialog: true,
          EditFormDialog: true,
          OrthographicDialog: true,
        },
      },
    })

    const tile = wrapper.find('.character-management-panel__tile')
    const tileCard = wrapper.getComponent(ProductRecordCard)
    expect(tileCard.props('as')).toBe('button')
    expect(tile.element.tagName).toBe('BUTTON')
    expect(tile.attributes('type')).toBe('button')
    expect(tile.attributes('aria-pressed')).toBe('false')

    await tile.trigger('click')

    expect(tile.attributes('aria-pressed')).toBe('true')
    expect(wrapper.find('.detail-name').text()).toBe('主角')
  })

  it('uses product chips for character tile status metadata', () => {
    const state = createState()
    state.characters.value[0] = {
      ...state.characters.value[0],
      enabled: false,
      forms: [
        { form_id: 'form-1', form_name: '常服', description: '', reference_image: '/tmp/form.png' },
        { form_id: 'form-2', form_name: '礼服', description: '', reference_image: '' },
      ],
    }

    const wrapper = mount(CharacterManagementPanel, {
      props: {
        bookId: 'book-1',
        characterManagement: {},
        state,
      },
      global: {
        stubs: {
          CharacterDetailPanel: characterDetailPanelStub,
          AddCharacterDialog: true,
          EditCharacterDialog: true,
          AddFormDialog: true,
          EditFormDialog: true,
          OrthographicDialog: true,
        },
      },
    })

    const chips = wrapper.getComponent(ProductChipList)
    expect(chips.props('items')).toEqual([
      { id: 'forms', label: '2 个形态', tone: 'primary' },
      { id: 'disabled', label: '禁用', tone: 'warning' },
    ])
  })

  it('uses the product section header contract for the character archive heading', () => {
    const wrapper = mount(CharacterManagementPanel, {
      props: {
        bookId: 'book-1',
        characterManagement: {},
        state: createState(),
      },
      global: {
        stubs: {
          CharacterDetailPanel: characterDetailPanelStub,
          AddCharacterDialog: true,
          EditCharacterDialog: true,
          AddFormDialog: true,
          EditFormDialog: true,
          OrthographicDialog: true,
        },
      },
    })
    const source = readFileSync(componentSourcePath, 'utf8')
    const header = wrapper.getComponent(ProductSectionHeader)

    expect(header.props()).toMatchObject({
      title: '角色档案',
      description: '点击角色查看和管理形态',
      iconName: 'users',
    })
    expect(header.text()).toContain('新增角色')
    expect(source).toContain("import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'")
    expect(source).not.toContain('class="section-header"')
    expect(source).not.toContain('class="section-title"')
    expect(source).not.toContain('.section-header {')
    expect(source).not.toContain('.section-title h4')
  })

  it('does not override shared button primitive variables at the panel root', () => {
    const source = readFileSync(componentSourcePath, 'utf8')
    const rootStyle = source.match(/\.character-management-panel \{(?<body>[\s\S]*?)\n\}/)

    expect(rootStyle?.groups?.body ?? '').not.toMatch(/--ui-button-/)
  })

  it('drives orthographic dialog status through typed props instead of child instance methods', () => {
    const source = readFileSync(componentSourcePath, 'utf8')

    expect(source).toContain(':is-generating="orthoGenerating"')
    expect(source).toContain(':result-image-path="orthoResultImagePath"')
    expect(source).not.toContain('orthoDialogRef')
    expect(source).not.toContain('setGenerating')
    expect(source).not.toContain('setResult')
  })

  it('lets the character list and detail pane stack based on the continuation workspace width', () => {
    const source = readFileSync(componentSourcePath, 'utf8')

    expect(source).toMatch(/\.character-management-panel \{[\s\S]*?container-type: inline-size;[\s\S]*?container-name: continuation-character-management;/)
    expect(source).toMatch(
      /@container continuation-character-management \(max-width: 640px\) \{[\s\S]*?\.character-management-panel__layout \{[\s\S]*?grid-template-columns: 1fr;[\s\S]*?\.character-management-panel__grid \{[\s\S]*?repeat\(auto-fill, minmax\(120px, 1fr\)\)[\s\S]*?max-height: none;/,
    )
  })

  it('renders the empty character list through product status feedback', () => {
    const state = createState()
    state.characters.value = []

    const wrapper = mount(CharacterManagementPanel, {
      props: {
        bookId: 'book-1',
        characterManagement: {},
        state,
      },
      global: {
        stubs: {
          CharacterDetailPanel: characterDetailPanelStub,
          AddCharacterDialog: true,
          EditCharacterDialog: true,
          AddFormDialog: true,
          EditFormDialog: true,
          OrthographicDialog: true,
        },
      },
    })

    const banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props('tone')).toBe('neutral')
    expect(banner.props('role')).toBe('note')
    expect(banner.props('iconName')).toBe('users')
    expect(wrapper.text()).toContain('点击“新增角色”添加')
    expect(wrapper.find('.empty-state').exists()).toBe(false)
  })

  it('uses the product avatar contract for character tiles', () => {
    const source = readFileSync(componentSourcePath, 'utf8')

    expect(source).toContain("import ProductAvatar from '@/components/product/ProductAvatar.vue'")
    expect(source).toContain('<ProductAvatar')
    expect(source).toContain('shape="rounded"')
    expect(source).not.toContain('class="tile-avatar"')
    expect(source).not.toContain('tile-avatar-placeholder')
    expect(source).not.toContain('char.name.charAt(0)')
  })

  it('uses product confirmation for destructive character form actions', async () => {
    const characterManagement = {
      deleteCharacter: vi.fn().mockResolvedValue(undefined),
      deleteForm: vi.fn().mockResolvedValue(undefined),
      deleteFormImage: vi.fn().mockResolvedValue(undefined),
    }
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const wrapper = mount(CharacterManagementPanel, {
      props: {
        bookId: 'book-1',
        characterManagement,
        state: createState(),
      },
      global: {
        stubs: {
          CharacterDetailPanel: characterDetailPanelStub,
          AddCharacterDialog: true,
          EditCharacterDialog: true,
          AddFormDialog: true,
          EditFormDialog: true,
          OrthographicDialog: true,
        },
      },
    })

    await wrapper.find('.character-management-panel__tile').trigger('click')
    await wrapper.get('.delete-form').trigger('click')
    await flushPromises()
    await wrapper.get('.delete-form-image').trigger('click')
    await flushPromises()
    await wrapper.get('.delete-character').trigger('click')
    await flushPromises()

    expect(confirmProductActionMock).toHaveBeenNthCalledWith(1, {
      title: '删除角色形态',
      message: '确定要删除形态"常服"吗？',
      confirmText: '删除',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(confirmProductActionMock).toHaveBeenNthCalledWith(2, {
      title: '删除形态参考图',
      message: '确定要删除形态参考图吗？',
      confirmText: '删除',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(confirmProductActionMock).toHaveBeenNthCalledWith(3, {
      title: '删除角色',
      message: '确定要删除角色"主角"吗？',
      confirmText: '删除',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(confirmSpy).not.toHaveBeenCalled()
    expect(characterManagement.deleteForm).toHaveBeenCalledWith('主角', 'form-1')
    expect(characterManagement.deleteFormImage).toHaveBeenCalledWith('主角', 'form-1')
    expect(characterManagement.deleteCharacter).toHaveBeenCalledWith('主角')
  })

  it('keeps character management panel hooks under the panel owner', () => {
    const source = readFileSync(componentSourcePath, 'utf8')

    for (const oldClass of [
      'character-empty-status',
      'character-panel-layout',
      'character-grid-panel',
      'character-tile',
      'tile-name',
      'tile-chips',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldClass}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldClass}\\b`))
    }
    expect(source).not.toContain('selected: selectedCharacter === char.name')
    expect(source).not.toContain('disabled: char.enabled === false')
    expect(source).not.toContain('.character-tile.selected')
    expect(source).not.toContain('.character-tile.disabled')

    for (const ownerClass of [
      'character-management-panel__empty-status',
      'character-management-panel__layout',
      'character-management-panel__grid',
      'character-management-panel__tile',
      'character-management-panel__tile--selected',
      'character-management-panel__tile--disabled',
      'character-management-panel__tile-name',
      'character-management-panel__tile-chips',
    ]) {
      expect(source).toContain(ownerClass)
    }
  })
})
