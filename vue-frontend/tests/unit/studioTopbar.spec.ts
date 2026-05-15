import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import StudioTopbar from '@/components/insight/studio/StudioTopbar.vue'

describe('StudioTopbar compact layout', () => {
  it('renders a compact title block without the large subtitle banner copy', () => {
    const wrapper = mount(StudioTopbar, {
      props: {
        subtitle: '编辑区优先，运行时预览收纳在右侧侧栏，适合长时间编卡。',
        bookTitle: '五等分',
        documentTitle: '上杉风太郎',
        documentOrigin: '分析生成',
        hasDocument: true,
        busy: true,
        busyLabel: '正在保存角色文档',
        savePending: false,
        validatePending: false,
      },
    })

    expect(wrapper.find('.title-row').exists()).toBe(true)
    expect(wrapper.find('.meta-row').exists()).toBe(true)
    expect(wrapper.text()).toContain('角色工坊 2.0')
    expect(wrapper.text()).toContain('当前书籍：五等分')
    expect(wrapper.text()).toContain('当前角色：上杉风太郎')
    expect(wrapper.text()).toContain('分析生成')
    expect(wrapper.text()).toContain('正在保存角色文档')
    expect(wrapper.text()).not.toContain('编辑区优先')
    expect(wrapper.text()).not.toContain('漫画分析 / 角色工坊')
  })
})
