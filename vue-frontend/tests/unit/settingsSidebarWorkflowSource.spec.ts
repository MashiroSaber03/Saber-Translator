import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('settings sidebar workflow property source contracts', () => {
  it('keeps workflow-button properties bound to the real sidebar owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/property/sidebarButtons.property.ts'), 'utf8')

    expect(source).toContain("from '@/components/translate/useSettingsSidebar'")
    expect(source).toContain("from '@/stores/imageStore'")

    for (const shadowContract of [
      'function hasCurrent' + 'Image',
      'function has' + 'Images',
      'function hasFailed' + 'Images',
      'function can' + 'Translate',
      'function supportsPage' + 'Selection',
      'function isRunWorkflow' + 'Disabled',
      'function calculateNavigation' + 'DisabledState',
      'interface Sidebar' + 'State',
      'interface Image' + 'State',
    ]) {
      expect(source).not.toContain(shadowContract)
    }

    for (const staleNarration of [
      '侧边栏工作流按钮禁用状态属性测试',
      '逻辑函数',
      '模拟组件',
      '生成器',
      'Property 48',
      '// ============================================================',
      '/' + '**',
      '验证',
      'return true',
    ]) {
      expect(source).not.toContain(staleNarration)
    }
  })
})
