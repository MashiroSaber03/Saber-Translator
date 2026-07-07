import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('translation state property source contract', () => {
  it('keeps translation-state properties on the current image-store contract', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'tests/property/translationState.property.ts'),
      'utf8',
    )

    for (const staleNarration of [
      '/' + '**',
      '翻译状态管理' + '属性测试',
      '使用 fast-check' + ' 进行属性基测试',
      '生成有效' + '的翻译状态',
      '生成有效' + '的图片数据',
      '每次迭代重新创建 Pinia',
      '验证' + '状态',
      'return ' + 'false',
      'localStorageMock',
      'Storage.prototype',
    ]) {
      expect(source).not.toContain(staleNarration)
    }

    expect(source).toContain('useImageStore')
    expect(source).toContain('expect(')
    expect(source).toContain('store.addImage(imageInput.fileName, imageInput.originalDataURL)')
    expect(source).not.toContain('store.addImage(imageData.originalDataURL, imageData.fileName)')
  })
})
