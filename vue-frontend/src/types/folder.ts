import type { ImageData } from './image'

/**
 * 文件夹树节点类型定义
 */
export interface FolderNode {
    /** 文件夹名称 */
    name: string
    /** 文件夹路径 */
    path: string
    /** 该文件夹下的图片 */
    images: ImageData[]
    /** 子文件夹 */
    subfolders: FolderNode[]
}
