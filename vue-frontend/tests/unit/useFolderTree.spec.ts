import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { ref } from 'vue'
import { useFolderTree } from '@/composables/useFolderTree'
import type { ImageData } from '@/types/image'

function createImage(id: string, fileName: string, folderPath = ''): ImageData {
  return {
    id,
    fileName,
    sourceAssetUrl: `/api/v2/assets/${id}`,
    translatedAssetUrl: null,
    cleanAssetUrl: null,
    bubbleStates: null,
    translationStatus: 'pending',
    translationFailed: false,
    fontSize: 24,
    autoFontSize: true,
    fontFamily: 'Arial',
    layoutDirection: 'auto',
    textColor: '#111111',
    fillColor: '#ffffff',
    inpaintMethod: 'lama',
    strokeEnabled: false,
    strokeColor: '#000000',
    strokeWidth: 0,
    hasUnsavedChanges: false,
    folderPath,
    relativePath: folderPath ? `${folderPath}/${fileName}` : fileName,
  }
}

describe('useFolderTree', () => {
  it('derives nested folders, breadcrumbs, and recursive image counts', () => {
    const images = ref([
      createImage('root-1', 'cover.png'),
      createImage('chapter-1', 'page-1.png', 'book/chapter-a'),
      createImage('chapter-2', 'page-2.png', 'book/chapter-a'),
      createImage('extra-1', 'extra.png', 'book/extras'),
    ])
    const folderTree = useFolderTree(images)

    expect(folderTree.useTreeMode.value).toBe(true)
    expect(folderTree.currentImages.value.map(image => image.fileName)).toEqual(['cover.png'])
    expect(folderTree.currentSubfolders.value.map(folder => folder.name)).toEqual(['book'])

    folderTree.enterFolder('book/chapter-a')

    expect(folderTree.breadcrumbs.value).toEqual([
      { name: '根目录', path: '' },
      { name: 'book', path: 'book' },
      { name: 'chapter-a', path: 'book/chapter-a' },
    ])
    expect(folderTree.currentImages.value.map(image => image.fileName)).toEqual(['page-1.png', 'page-2.png'])

    folderTree.goUp()
    expect(folderTree.currentFolderPath.value).toBe('book')
    expect(folderTree.getFolderImageCount(folderTree.folderTree.value!.subfolders[0]!)).toBe(3)

    folderTree.resetToRoot()
    expect(folderTree.currentFolderPath.value).toBe('')
  })

  it('keeps folder-tree source comments focused on current behavior contracts', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/composables/useFolderTree.ts'), 'utf8')

    for (const staleNarration of [
      '文件夹树逻辑封装',
      '// ============================================================',
      '计算属性',
      '当前浏览的文件夹路径',
      '路径映射缓存',
      '确保文件夹节点存在',
      '添加图片到对应文件夹',
      '按路径查找文件夹',
      '跳转到指定路径',
      '@param',
    ]) {
      expect(source).not.toContain(staleNarration)
    }
  })
})
