import { computed, ref, type Ref } from 'vue'
import type { ImageData } from '@/types/image'
import type { FolderNode } from '@/types/folder'

export function useFolderTree(images: Ref<ImageData[]>) {
  const currentFolderPath = ref<string>('')

  const useTreeMode = computed(() => images.value.some(image => image.folderPath))

  const folderTree = computed((): FolderNode | null => {
    if (!useTreeMode.value) return null

    const root: FolderNode = {
      name: '根目录',
      path: '',
      images: [],
      subfolders: [],
    }
    const folderMap = new Map<string, FolderNode>([['', root]])

    for (const image of images.value) {
      const folderPath = image.folderPath || ''

      if (folderPath && !folderMap.has(folderPath)) {
        let currentPath = ''

        for (const part of folderPath.split('/')) {
          const parentPath = currentPath
          currentPath = currentPath ? `${currentPath}/${part}` : part

          if (!folderMap.has(currentPath)) {
            const folder: FolderNode = {
              name: part,
              path: currentPath,
              images: [],
              subfolders: [],
            }
            folderMap.set(currentPath, folder)
            folderMap.get(parentPath)?.subfolders.push(folder)
          }
        }
      }

      folderMap.get(folderPath)?.images.push(image)
    }

    return root
  })

  const breadcrumbs = computed(() => {
    if (!currentFolderPath.value) {
      return [{ name: '根目录', path: '' }]
    }

    let path = ''
    return [
      { name: '根目录', path: '' },
      ...currentFolderPath.value.split('/').map(part => {
        path = path ? `${path}/${part}` : part
        return { name: part, path }
      }),
    ]
  })

  const currentFolder = computed((): FolderNode | null => {
    if (!folderTree.value) return null
    if (!currentFolderPath.value) return folderTree.value

    return findFolder(folderTree.value, currentFolderPath.value)
  })

  const currentSubfolders = computed(() => currentFolder.value?.subfolders || [])
  const currentImages = computed(() => currentFolder.value?.images || [])

  function enterFolder(folderPath: string): void {
    currentFolderPath.value = folderPath
  }

  function goUp(): void {
    if (!currentFolderPath.value) return

    const lastSlash = currentFolderPath.value.lastIndexOf('/')
    currentFolderPath.value = lastSlash > 0
      ? currentFolderPath.value.substring(0, lastSlash)
      : ''
  }

  function navigateTo(path: string): void {
    currentFolderPath.value = path
  }

  function getFolderImageCount(folder: FolderNode): number {
    return folder.images.length + folder.subfolders.reduce(
      (count, subfolder) => count + getFolderImageCount(subfolder),
      0
    )
  }

  function resetToRoot(): void {
    currentFolderPath.value = ''
  }

  return {
    currentFolderPath,
    useTreeMode,
    folderTree,
    breadcrumbs,
    currentSubfolders,
    currentImages,
    enterFolder,
    goUp,
    navigateTo,
    getFolderImageCount,
    resetToRoot,
  }
}

function findFolder(node: FolderNode, targetPath: string): FolderNode | null {
  if (node.path === targetPath) return node

  for (const subfolder of node.subfolders) {
    const found = findFolder(subfolder, targetPath)
    if (found) return found
  }

  return null
}
