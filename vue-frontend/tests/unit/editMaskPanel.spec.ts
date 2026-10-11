import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import EditMaskPanel from '@/components/edit/EditMaskPanel.vue'
import { maskVisibility } from '@/utils/maskPreview'

vi.mock('@/composables/usePublicUserAccess', () => ({
  usePublicUserAccess: () => ({ modelOptions: (options: unknown[]) => options }),
}))
function mountPanel() {
  return mount(EditMaskPanel, {
    props: {
      tool: null, visibility: maskVisibility(), preview: null,
      busy: false, loading: false, saving: false, applying: false, dirty: false,
      error: '', inpaintMethod: 'solid', fillColor: '#ffffff', brushSize: 20,
    },
    global: { stubs: { UiSelect: true, UiColorInput: true } },
  })
}
describe('mask layer controls', () => {
  it('keeps every layer switch visible in the toolbar and changes display independently', async () => {
    const wrapper = mountPanel()
    expect(wrapper.findAll('[aria-label^="显示"]')).toHaveLength(6)
    expect(wrapper.find('[aria-label="显示文本框范围"]').exists()).toBe(true)
    expect(wrapper.find('[aria-label="显示文本框筛选"]').exists()).toBe(false)
    expect(wrapper.find('[role="dialog"]').exists()).toBe(false)
    expect(wrapper.find('[aria-label="图层显示开关"]').exists()).toBe(false)
    expect(wrapper.get('[aria-label="显示原始文字范围"]').attributes('disabled')).toBeDefined()
    expect(wrapper.get('[aria-label="显示原始文字范围"]').attributes('title')).toContain('当前页没有精确文字范围')
    await wrapper.get('[aria-label="显示自动生效区域"]').trigger('click')
    const automatic = { ...maskVisibility(), automatic: true }
    expect(wrapper.emitted('visibility')?.[0]).toEqual([automatic])
    await wrapper.setProps({ visibility: automatic })
    await wrapper.get('[aria-label="显示手动保留区域"]').trigger('click')
    expect(wrapper.emitted('visibility')?.[1]).toEqual([{ ...automatic, preserve: true }])
    expect(wrapper.get('[aria-label="显示最终修复范围"]').attributes('aria-pressed')).toBe('true')
    expect(wrapper.emitted('tool')).toBeUndefined()
    expect(wrapper.emitted('apply')).toBeUndefined()
    wrapper.unmount()
  })
  it('uses labeled icon-only brushes without a redundant pan control', async () => {
    const wrapper = mountPanel()
    const group = wrapper.get('[aria-label="修复范围画笔"]')
    expect(group.findAll('button')).toHaveLength(3)
    expect(wrapper.find('[aria-label="移动与缩放"]').exists()).toBe(false)
    for (const button of group.findAll('button')) {
      expect(button.text()).toBe('')
      expect(button.attributes('title')).toBeTruthy()
      expect(button.attributes('aria-pressed')).toBe('false')
    }
    await group.get('[aria-label="增加修复区域 (R)"]').trigger('click')
    expect(wrapper.emitted('tool')?.[0]).toEqual(['repair'])
    await wrapper.setProps({ tool: 'repair' })
    expect(group.get('[aria-label="增加修复区域 (R)"]').attributes('aria-pressed')).toBe('true')
    await group.get('[aria-label="增加修复区域 (R)"]').trigger('click')
    expect(wrapper.emitted('tool')?.[1]).toEqual(['repair']) // Workspace toggles the same tool off.
    wrapper.unmount()
  })
  it('describes actual clipping and processing beside the permanent layer chain', async () => {
    const wrapper = mountPanel()
    await wrapper.setProps({ preview: {
      pageId: 'page', sourceRevision: 1, documentRevision: 1, width: 4, height: 4,
      preciseMask: 'png', boxMask: 'png', automaticMask: 'png', userMask: 'png', finalMask: 'png',
      processing: { boxExpandRatio: 2, dilateSize: 4, closeGaps: true },
    } })
    expect(wrapper.get('[aria-label="显示原始文字范围"]').attributes('disabled')).toBeUndefined()
    expect(wrapper.get('[aria-label="显示文本框范围"]').attributes('title')).toContain('扩大 2%')
    expect(wrapper.text()).toContain('框扩大 2% · 膨胀 4px · 闭合空隙')
    expect(wrapper.get('[aria-label="查看修复范围图层"]').text()).toMatch(/原始.*→.*文本框范围.*→.*自动.*\+.*手动修复.*−.*手动保留.*=.*最终/s)
    expect(wrapper.find('[aria-label="文本框选区图例"]').exists()).toBe(false)
    await wrapper.get('[aria-label="显示文本框范围"]').trigger('click')
    expect(wrapper.emitted('visibility')?.[0]).toEqual([{ ...maskVisibility(), boxes: true }])
    await wrapper.setProps({ visibility: { ...maskVisibility(), boxes: true } })
    expect(wrapper.find('[aria-label="文本框选区图例"]').exists()).toBe(false)
    expect(wrapper.get('[aria-label="显示文本框范围"]').attributes('title')).toContain('黄色虚线和淡黄色底色')
    expect(wrapper.text()).not.toMatch(/框内保留的文字|框外排除的文字/)
    await wrapper.setProps({ preview: { ...wrapper.props('preview')!, preciseMask: null } })
    expect(wrapper.get('[aria-label="显示文本框范围"]').text()).toBe('文本框范围')
    expect(wrapper.get('[aria-label="显示文本框范围"]').attributes('disabled')).toBeUndefined()
    await wrapper.setProps({ visibility: maskVisibility() })
    expect(wrapper.find('[aria-label="文本框选区图例"]').exists()).toBe(false)
    wrapper.unmount()
  })
  it('keeps repair available when every overlay is hidden', async () => {
    const wrapper = mountPanel()
    const hidden = maskVisibility()
    hidden.final = false
    await wrapper.setProps({ visibility: hidden })
    expect(wrapper.text()).toContain('覆盖层已隐藏，修复范围不变')
    expect(wrapper.get('[aria-label="按当前范围修复"]').attributes('disabled')).toBeUndefined()
    await wrapper.get('[aria-label="按当前范围修复"]').trigger('click')
    expect(wrapper.emitted('apply')).toHaveLength(1)
    wrapper.unmount()
  })
})
