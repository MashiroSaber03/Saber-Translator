import { ref, computed, type Ref } from 'vue'
import type { ImageData } from '@/types/image'
import type { FolderNode } from '@/types/folder'

/**
 * 文件夹树逻辑封装（面包屑导航模式）
 * @param images 图片列表响应式对象
 */
export function useFolderTree(images: Ref<ImageData[]>) {
    // ============================================================
    // 状态
    // ============================================================

    /** 当前浏览的文件夹路径 */
    const currentFolderPath = ref<string>('')

    // ============================================================
    // 计算属性
    // ============================================================

    /**
     * 是否使用树形模式
     */
    const useTreeMode = computed(() => {
        return images.value.some(img => img.folderPath)
    })

    /**
     * 构建完整的文件夹树结构
     */
    const folderTree = computed((): FolderNode | null => {
        if (!useTreeMode.value) return null

        const root: FolderNode = {
            name: '根目录',
            path: '',
            isExpanded: true,
            images: [],
            subfolders: []
        }

        // 路径映射缓存
        const folderMap = new Map<string, FolderNode>()
        folderMap.set('', root)

        for (const image of images.value) {
            const folderPath = image.folderPath || ''

            // 确保文件夹节点存在
            if (folderPath && !folderMap.has(folderPath)) {
                const pathParts = folderPath.split('/')
                let currentPath = ''

                for (const part of pathParts) {
                    const prevPath = currentPath
                    currentPath = currentPath ? `${currentPath}/${part}` : part

                    if (!folderMap.has(currentPath)) {
                        const newFolder: FolderNode = {
                            name: part,
                            path: currentPath,
                            isExpanded: true,
                            images: [],
                            subfolders: []
                        }
                        folderMap.set(currentPath, newFolder)
                        if (folderMap.has(prevPath)) {
                            folderMap.get(prevPath)!.subfolders.push(newFolder)
                        }
                    }
                }
            }

            // 添加图片到对应文件夹
            if (folderMap.has(folderPath)) {
                folderMap.get(folderPath)!.images.push(image)
            }
        }

        return root
    })

    /**
     * 面包屑路径数组
     */
    const breadcrumbs = computed(() => {
        if (!currentFolderPath.value) {
            return [{ name: '根目录', path: '' }]
        }

        const parts = currentFolderPath.value.split('/')
        const crumbs = [{ name: '根目录', path: '' }]
        let path = ''

        for (const part of parts) {
            path = path ? `${path}/${part}` : part
            crumbs.push({ name: part, path })
        }

        return crumbs
    })

    /**
     * 当前文件夹节点
     */
    const currentFolder = computed((): FolderNode | null => {
        if (!folderTree.value) return null

        if (!currentFolderPath.value) {
            return folderTree.value
        }

        // 按路径查找文件夹
        const findFolder = (node: FolderNode, targetPath: string): FolderNode | null => {
            if (node.path === targetPath) return node
            for (const sub of node.subfolders) {
                const found = findFolder(sub, targetPath)
                if (found) return found
            }
            return null
        }

        return findFolder(folderTree.value, currentFolderPath.value)
    })

    /**
     * 当前文件夹的直接子文件夹
     */
    const currentSubfolders = computed(() => {
        return currentFolder.value?.subfolders || []
    })

    /**
     * 当前文件夹的直接图片
     */
    const currentImages = computed(() => {
        return currentFolder.value?.images || []
    })

    // ============================================================
    // 方法
    // ============================================================

    /**
     * 进入文件夹
     */
    function enterFolder(folderPath: string) {
        currentFolderPath.value = folderPath
    }

    /**
     * 返回上级文件夹
     */
    function goUp() {
        if (!currentFolderPath.value) return
        const lastSlash = currentFolderPath.value.lastIndexOf('/')
        currentFolderPath.value = lastSlash > 0
            ? currentFolderPath.value.substring(0, lastSlash)
            : ''
    }

    /**
     * 跳转到指定路径（用于面包屑点击）
     */
    function navigateTo(path: string) {
        currentFolderPath.value = path
    }

    /**
     * 获取文件夹内图片数量（包括子文件夹）
     */
    function getFolderImageCount(folder: FolderNode): number {
        let count = folder.images.length
        for (const subfolder of folder.subfolders) {
            count += getFolderImageCount(subfolder)
        }
        return count
    }

    /**
     * 重置到根目录
     */
    function resetToRoot() {
        currentFolderPath.value = ''
    }

    return {
        currentFolderPath,
        useTreeMode,
        folderTree,
        breadcrumbs,
        currentFolder,
        currentSubfolders,
        currentImages,
        enterFolder,
        goUp,
        navigateTo,
        getFolderImageCount,
        resetToRoot
    }
}
