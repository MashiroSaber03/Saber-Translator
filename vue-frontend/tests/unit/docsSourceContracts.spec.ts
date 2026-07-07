import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('documentation source contracts', () => {
  it('keeps the current coding style guide aligned with product header and form primitives', () => {
    const codingStyle = source('CODING_STYLE.md')

    expect(codingStyle).toContain('ProductPageHeader')
    expect(codingStyle).toContain('ProductHeaderAction')
    expect(codingStyle).toContain('UiFormGrid')
    expect(codingStyle).toContain('UiSelect')
    expect(codingStyle).toContain('UiNumberField')
    expect(codingStyle).not.toContain('UiModalSection')
    expect(codingStyle).not.toContain('AppHeader')
  })

  it('keeps root frontend config files free of scaffold narration', () => {
    const configFiles = [
      'eslint.config.js',
      'vite.config.ts',
      'vitest.config.ts',
      'index.html',
      '.stylelintignore',
      'tsconfig.app.json',
      'tsconfig.node.json',
    ]
    const staleNarration = [
      '基础 JavaScript 规则',
      'TypeScript 规则',
      'Vue 规则',
      '自定义配置',
      '添加浏览器全局变量',
      '通用规则',
      '测试文件配置',
      'Node 脚本配置',
      '忽略文件',
      '路径别名配置',
      '构建配置 - 输出到 Flask 静态目录',
      '生产构建默认不输出 sourcemap',
      '代码分割配置',
      '手动分割代码块',
      'Vue 核心库',
      '工具库',
      '设置 chunk 大小警告阈值',
      '基础路径',
      '开发服务器配置',
      '允许局域网访问',
      '所有 API 请求代理到 Flask',
      '注意：静态资源',
      '预览服务器配置',
      '使用 jsdom 作为测试环境',
      '全局导入测试函数',
      '测试文件匹配模式',
      '属性测试配置',
      '覆盖率配置',
      '全局样式由',
      'Stylelint Ignore',
      '路径别名',
      '严格模式 - TypeScript 类型检查',
      'Bundler mode',
      'Linting',
    ]

    for (const file of configFiles) {
      const content = source(file)
      for (const text of staleNarration) {
        expect(content, file).not.toContain(text)
      }
    }
  })
})
