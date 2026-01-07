import type { ImageData } from './image'

/**
 * 文件夹树节点类型定义
 */
export interface FolderNode {
    /** 文件夹名称 */
    name: string
    /** 文件夹路径 */
    path: string
    /** 是否展开 */
    isExpanded: boolean
    /** 该文件夹下的图片 */
    images: ImageData[]
    /** 子文件夹 */
    subfolders: FolderNode[]
}

/**
 * 文件夹树上下文（用于 provide/inject）
 */
export interface FolderTreeContext {
    getImageGlobalIndex: (image: ImageData) => number
    getStatusType: (image: ImageData) => 'failed' | 'labeled' | 'processing' | null
    toggleFolder: (folderPath: string) => void
    folderExpandState: Record<string, boolean>
    currentIndex: number
}

export const FOLDER_TREE_CONTEXT_KEY = Symbol('folderTreeContext')
