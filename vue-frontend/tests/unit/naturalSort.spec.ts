import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { naturalSort, naturalSortCompare, naturalSortKey } from '@/utils'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('naturalSort utility', () => {
  it('sorts mixed chapter paths by numeric segments without mutating the input', () => {
    const files = [
      { path: '第10话\\002.jpg' },
      { path: '第2话\\010.jpg' },
      { path: '第2话\\002.jpg' },
      { path: '第1话\\001.jpg' },
    ]

    const sorted = naturalSort(files, file => file.path)

    expect(sorted.map(file => file.path)).toEqual([
      '第1话\\001.jpg',
      '第2话\\002.jpg',
      '第2话\\010.jpg',
      '第10话\\002.jpg',
    ])
    expect(files.map(file => file.path)).toEqual([
      '第10话\\002.jpg',
      '第2话\\010.jpg',
      '第2话\\002.jpg',
      '第1话\\001.jpg',
    ])
    expect(naturalSortCompare('file2.jpg', 'file10.jpg')).toBeLessThan(0)
    expect(naturalSortKey('Folder\\Page10.png')).toEqual([
      [true, 'folder/page'],
      [false, 10],
      [true, '.png'],
    ])
  })

  it('keeps the source compact and free of tutorial narration', () => {
    const content = source('src/utils/naturalSort.ts')

    for (const staleNarration of [
      '/**',
      '@param',
      '@returns',
      '自然排序工具函数',
      '实现效果',
      '规范化路径分隔符',
      '添加数字前的文本部分',
      '添加最后的文本部分',
      '同类型比较',
      '长度不同时',
    ]) {
      expect(content).not.toContain(staleNarration)
    }
  })

  it('keeps image upload property coverage on the shared utility contract', () => {
    const content = source('tests/property/imageUpload.property.ts')

    expect(content).toContain("import { naturalSort, naturalSortCompare } from '@/utils'")
    expect(content).not.toContain('function naturalSort(')
    expect(content).not.toContain('Property 26.')
    expect(content).not.toContain('与 ImageUpload 组件中的实现一致')
  })
})
