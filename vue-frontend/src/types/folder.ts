import type { ImageData } from './image'

export interface FolderNode {
  name: string
  path: string
  images: ImageData[]
  subfolders: FolderNode[]
}
