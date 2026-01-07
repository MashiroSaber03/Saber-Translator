# 文件夹导入功能实现方案

## 📋 需求概述

为 Saber-Translator 添加文件夹导入功能，允许用户：
- 选择本地文件夹，自动导入其中所有图片
- 递归扫描子文件夹中的图片
- 按自然排序（1, 2, 10 而非 1, 10, 2）加载图片
- 在缩略图区域按文件夹分组显示图片
- 支持折叠/展开子文件夹

---

## 🎯 参考项目

参考 [manga-translator-ui](https://github.com/hgmzhn/manga-translator-ui) 的文件夹导入逻辑：
- `desktop_qt_ui/services/file_service.py` - 文件夹扫描、自然排序
- `desktop_qt_ui/editor/editor_logic.py` - 文件夹添加逻辑
- `desktop_qt_ui/widgets/file_list_view.py` - 树形文件夹显示

---

## 📁 修改文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `vue-frontend/src/utils/naturalSort.ts` | **新增** | 自然排序工具函数 |
| `vue-frontend/src/utils/index.ts` | 修改 | 导出 naturalSort |
| `vue-frontend/src/types/image.ts` | 修改 | 添加 `relativePath` 和 `folderPath` 字段 |
| `vue-frontend/src/types/folder.ts` | **新增** | 定义 FolderNode 类型和 Context |
| `vue-frontend/src/types/index.ts` | 修改 | 导出 folder 类型 |
| `vue-frontend/src/composables/useFolderTree.ts` | **新增** | 文件夹树逻辑 Composable |
| `vue-frontend/src/composables/index.ts` | 修改 | 导出 useFolderTree |
| `vue-frontend/src/components/translate/ImageUpload.vue` | 修改 | 添加文件夹选择和路径传递 |
| `vue-frontend/src/components/translate/ThumbnailSidebar.vue` | **重构** | 支持双模式渲染 |
| `vue-frontend/src/components/translate/FolderTreeNode.vue` | **新增** | 递归文件夹树节点组件 |

**共 10 个文件（4 个新增，6 个修改）**

---

## 🔧 详细修改方案

### 1. 新增自然排序工具函数

**文件**: `vue-frontend/src/utils/naturalSort.ts`

```typescript
/**
 * 自然排序工具函数
 * 
 * 实现效果：
 * - file1.jpg, file2.jpg, file10.jpg → 按 1, 2, 10 排序
 * - 第1话/001.jpg, 第2话/001.jpg, 第10话/001.jpg → 按 1, 2, 10 排序
 * 
 * 参考: manga-translator-ui/file_service.py 的 _natural_sort_key
 */

/**
 * 生成自然排序的键
 * @param path 文件路径或文件名
 * @returns 用于排序的键数组
 */
export function naturalSortKey(path: string): Array<[boolean, number | string]> {
  // 规范化路径分隔符
  const normalizedPath = path.replace(/\\/g, '/')
  
  // 将路径分割成文本和数字部分
  const parts: Array<[boolean, number | string]> = []
  const regex = /(\d+)/g
  let lastIndex = 0
  let match: RegExpExecArray | null
  
  while ((match = regex.exec(normalizedPath)) !== null) {
    // 添加数字前的文本部分
    if (match.index > lastIndex) {
      const textPart = normalizedPath.slice(lastIndex, match.index)
      if (textPart) {
        parts.push([true, textPart.toLowerCase()])
      }
    }
    // 添加数字部分
    parts.push([false, parseInt(match[0], 10)])
    lastIndex = regex.lastIndex
  }
  
  // 添加最后的文本部分
  if (lastIndex < normalizedPath.length) {
    const textPart = normalizedPath.slice(lastIndex)
    if (textPart) {
      parts.push([true, textPart.toLowerCase()])
    }
  }
  
  return parts
}

/**
 * 自然排序比较函数
 * @param a 第一个路径
 * @param b 第二个路径
 * @returns 比较结果 (-1, 0, 1)
 */
export function naturalSortCompare(a: string, b: string): number {
  const keyA = naturalSortKey(a)
  const keyB = naturalSortKey(b)
  
  const minLength = Math.min(keyA.length, keyB.length)
  
  for (let i = 0; i < minLength; i++) {
    const [isTextA, valA] = keyA[i]
    const [isTextB, valB] = keyB[i]
    
    // 如果类型不同：数字排在文本前面
    if (isTextA !== isTextB) {
      return isTextA ? 1 : -1
    }
    
    // 同类型比较
    if (valA < valB) return -1
    if (valA > valB) return 1
  }
  
  // 长度不同时，短的排前面
  return keyA.length - keyB.length
}

/**
 * 对文件列表进行自然排序
 * @param files 文件数组
 * @param getPath 获取排序路径的函数（可选）
 * @returns 排序后的数组（原数组不变）
 */
export function naturalSort<T>(
  files: T[],
  getPath: (item: T) => string = (item) => String(item)
): T[] {
  return [...files].sort((a, b) => naturalSortCompare(getPath(a), getPath(b)))
}
```

---

### 2. 更新 utils/index.ts

**文件**: `vue-frontend/src/utils/index.ts`

在文件末尾添加导出：

```typescript
// 自然排序工具函数
export {
  naturalSortKey,
  naturalSortCompare,
  naturalSort
} from './naturalSort'
```

---

### 3. 修改图片类型定义

**文件**: `vue-frontend/src/types/image.ts`

在 `ImageData` 接口中添加字段：

```typescript
export interface ImageData {
  // ... 现有字段 ...
  
  /** 文件的原始路径（用于文件夹分组） */
  relativePath?: string
  
  /** 所属文件夹路径 */
  folderPath?: string
}
```

---

### 4. 新增文件夹类型定义

**文件**: `vue-frontend/src/types/folder.ts`

```typescript
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
```

---

### 5. 更新 types/index.ts

**文件**: `vue-frontend/src/types/index.ts`

添加导出：

```typescript
// 文件夹类型
export * from './folder'
```

---

### 6. 新增文件夹树逻辑 Composable

**文件**: `vue-frontend/src/composables/useFolderTree.ts`

```typescript
import { ref, computed, type Ref } from 'vue'
import type { ImageData } from '@/types/image'
import type { FolderNode } from '@/types/folder'

/**
 * 文件夹树逻辑封装
 * @param images 图片列表响应式对象
 */
export function useFolderTree(images: Ref<ImageData[]>) {
  // ============================================================
  // 状态
  // ============================================================
  
  /** 文件夹展开状态 */
  const folderExpandState = ref<Record<string, boolean>>({})
  
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
   * 构建文件夹树结构
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
    
    // 简单的路径映射缓存
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
              isExpanded: folderExpandState.value[currentPath] ?? true,
              images: [],
              subfolders: []
            }
            folderMap.set(currentPath, newFolder)
            // 将新文件夹添加到父文件夹的子列表中
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
  
  // ============================================================
  // 方法
  // ============================================================
  
  /**
   * 切换文件夹展开状态
   */
  function toggleFolder(folderPath: string) {
    folderExpandState.value[folderPath] = 
      !(folderExpandState.value[folderPath] ?? true)
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
  
  return {
    folderExpandState,
    useTreeMode,
    folderTree,
    toggleFolder,
    getFolderImageCount
  }
}
```

---

### 7. 更新 composables/index.ts

**文件**: `vue-frontend/src/composables/index.ts`

添加导出：

```typescript
export { useFolderTree } from './useFolderTree'
```

---

### 8. 修改 ImageUpload.vue

**文件**: `vue-frontend/src/components/translate/ImageUpload.vue`

#### 8.1 添加导入

在 `<script setup lang="ts">` 开头添加:

```typescript
import { naturalSort } from '@/utils'
```

#### 8.2 添加文件夹输入框引用

在状态定义区域添加:

```typescript
/** 文件夹输入框引用 */
const folderInputRef = ref<HTMLInputElement | null>(null)
```

#### 8.3 添加触发文件夹选择方法

```typescript
/**
 * 触发文件夹选择对话框
 */
function triggerFolderSelect() {
  folderInputRef.value?.click()
}
```

#### 8.4 添加处理文件夹选择方法

```typescript
/**
 * 处理文件夹选择
 */
async function handleFolderSelect(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files || input.files.length === 0) return

  const allFiles = Array.from(input.files)
  const imageFiles = allFiles.filter(file => file.type.startsWith('image/'))

  if (imageFiles.length === 0) {
    showToast('所选文件夹中没有找到图片文件', 'warning')
    input.value = ''
    return
  }

  // 按相对路径进行自然排序
  const sortedFiles = naturalSort(imageFiles, (file) => file.webkitRelativePath)
  
  console.log(`从文件夹导入 ${sortedFiles.length} 张图片`)
  
  // 处理文件并保留文件夹信息
  await processFilesWithFolderInfo(sortedFiles)
  
  input.value = ''
}

/**
 * 处理文件并保留文件夹信息
 */
async function processFilesWithFolderInfo(files: File[]) {
  if (files.length === 0) return
  
  isLoading.value = true
  showProgress.value = true
  uploadProgress.value = 0
  
  try {
    let processedCount = 0
    const totalFiles = files.length
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      if (!file || !file.type.startsWith('image/')) continue
      
      currentFileName.value = file.name
      
      // 获取相对路径信息
      const relativePath = file.webkitRelativePath || ''
      // 提取文件夹路径（去掉文件名）
      const folderPath = relativePath.includes('/')
        ? relativePath.substring(0, relativePath.lastIndexOf('/'))
        : ''
      
      // 读取图片并添加
      await new Promise<void>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = (e) => {
          const dataURL = e.target?.result as string
          // 使用带文件夹信息的方式添加
          imageStore.addImage(file.name, dataURL, {
            relativePath,
            folderPath
          })
          resolve()
        }
        reader.onerror = () => reject(new Error(`读取图片失败: ${file.name}`))
        reader.readAsDataURL(file)
      })
      
      processedCount++
      uploadProgress.value = Math.round(((i + 1) / totalFiles) * 100)
    }
    
    if (processedCount > 0) {
      showToast(`已添加 ${processedCount} 张图片`, 'success')
      emit('uploadComplete', processedCount)
    }
  } catch (error) {
    console.error('处理文件失败:', error)
    const errMsg = error instanceof Error ? error.message : '处理文件失败'
    showToast(errMsg, 'error')
  } finally {
    isLoading.value = false
    showProgress.value = false
  }
}
```

#### 8.5 更新 defineExpose

```typescript
defineExpose({
  triggerFileSelect,
  triggerFolderSelect,  // 新增
  processFiles,
  clearError,
})
```

#### 8.6 修改模板 - 添加文件夹按钮

```html
<div class="drop-content">
  <p class="drop-text">
    拖拽图片、PDF或MOBI文件到这里，或 
    <span class="select-link" @click="triggerFileSelect">
      选择文件
    </span>
    <span class="separator"> | </span>
    <span class="select-link folder-link" @click="triggerFolderSelect">
      📁 选择文件夹
    </span>
    <span class="separator"> | </span>
    <span class="select-link web-import-link" @click="triggerWebImport">
      🌐 从网页导入
    </span>
  </p>
</div>
```

#### 8.7 添加隐藏的文件夹输入框

在现有的 `<input ref="fileInputRef" ...>` 后面添加:

```html
<!-- 隐藏的文件夹输入框 -->
<input 
  ref="folderInputRef"
  type="file" 
  webkitdirectory
  class="file-input"
  @change="handleFolderSelect"
>
```

#### 8.8 添加样式

```css
.folder-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
```

---

### 9. 重构 ThumbnailSidebar.vue

**文件**: `vue-frontend/src/components/translate/ThumbnailSidebar.vue`

```vue
<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted, provide } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useFolderTree } from '@/composables/useFolderTree'
import { FOLDER_TREE_CONTEXT_KEY, type FolderTreeContext } from '@/types/folder'
import type { ImageData } from '@/types/image'
import FolderTreeNode from './FolderTreeNode.vue'

// ... Props/Emits ...

const imageStore = useImageStore()
const images = computed(() => imageStore.images)
const currentIndex = computed(() => imageStore.currentImageIndex)
const hasImages = computed(() => imageStore.hasImages)

// 1. 使用 Composable 管理树逻辑
const { 
  folderExpandState, 
  useTreeMode, 
  folderTree, 
  toggleFolder 
} = useFolderTree(images)

// 2. 辅助方法
function getImageGlobalIndex(image: ImageData): number {
  return images.value.findIndex(img => img.id === image.id)
}

function getStatusType(image: ImageData): 'failed' | 'labeled' | 'processing' | null {
  if (image.translationFailed) return 'failed'
  if (image.isManuallyAnnotated) return 'labeled'
  if (image.translationStatus === 'processing') return 'processing'
  return null
}

// 3. Provide 上下文给子组件
// 注意：直接传递对象，函数引用是稳定的，而响应式值通过闭包获取
const folderTreeContext: FolderTreeContext = {
  getImageGlobalIndex,
  getStatusType,
  toggleFolder,
  get folderExpandState() { return folderExpandState.value },
  get currentIndex() { return currentIndex.value }
}
provide(FOLDER_TREE_CONTEXT_KEY, folderTreeContext)

// 保留现有方法（不修改）:
// - containerRef, thumbnailRefs
// - setThumbnailRef()
// - scrollToActiveThumbnail()
// - getThumbnailClasses()
// - getThumbnailTitle()
// - watch(currentIndex, ...)
// - onMounted()

// 4. 事件处理
function handleClick(index: number) {
  emit('select', index)
}
</script>

<template>
  <aside id="thumbnail-sidebar" class="thumbnail-sidebar">
    <div class="card thumbnail-card">
      <h2>图片概览</h2>
      
      <!-- 树形模式 -->
      <div 
        v-if="hasImages && useTreeMode && folderTree"
        ref="containerRef"
        class="thumbnail-tree"
      >
        <FolderTreeNode
          v-for="subfolder in folderTree.subfolders"
          :key="subfolder.path"
          :folder="subfolder"
          @select="handleClick"
          @set-ref="setThumbnailRef"
        />
        <!-- 根目录图片 -->
        <div
          v-for="image in folderTree.images"
          :key="image.id"
          class="thumbnail-item"
          :class="{ active: getImageGlobalIndex(image) === currentIndex }"
          @click="handleClick(getImageGlobalIndex(image))"
          :ref="(el) => setThumbnailRef(el as HTMLElement | null, getImageGlobalIndex(image))"
        >
          <img 
            v-if="image.originalDataURL"
            :src="image.originalDataURL" 
            class="thumbnail-image"
          >
          <span v-if="getStatusType(image) === 'failed'" class="translation-failed-indicator">!</span>
          <span v-else-if="getStatusType(image) === 'labeled'" class="labeled-indicator">✏️</span>
          <div v-if="getStatusType(image) === 'processing'" class="thumbnail-processing-indicator">⟳</div>
        </div>
      </div>
      
      <!-- 扁平模式（保留原有完整代码） -->
      <ul 
        v-else-if="hasImages"
        ref="containerRef"
        id="thumbnailList"
        class="thumbnail-list"
      >
        <li
          v-for="(image, index) in images"
          :key="image.id"
          :ref="(el) => setThumbnailRef(el as HTMLElement | null, index)"
          class="thumbnail-item"
          :class="[
            { active: index === currentIndex },
            ...getThumbnailClasses(image)
          ]"
          :title="getThumbnailTitle(image)"
          @click="handleClick(index)"
        >
          <img 
            v-if="image.originalDataURL"
            :src="image.originalDataURL" 
            :alt="image.fileName"
            class="thumbnail-image"
          >
          <span v-if="getStatusType(image) === 'failed'" class="translation-failed-indicator">!</span>
          <span v-else-if="getStatusType(image) === 'labeled'" class="labeled-indicator">✏️</span>
          <div v-if="getStatusType(image) === 'processing'" class="thumbnail-processing-indicator">⟳</div>
        </li>
      </ul>
      
      <div v-else class="empty-state">
        <p>暂无图片</p>
      </div>
      
    </div>
  </aside>
</template>
```

---

### 10. 新增 FolderTreeNode.vue

**文件**: `vue-frontend/src/components/translate/FolderTreeNode.vue`

```vue
<script setup lang="ts">
import { computed, inject } from 'vue'
import type { FolderNode, FolderTreeContext } from '@/types/folder'
import { FOLDER_TREE_CONTEXT_KEY } from '@/types/folder'

const props = defineProps<{
  folder: FolderNode
}>()

const emit = defineEmits<{
  (e: 'select', index: number): void
  (e: 'setRef', el: HTMLElement | null, index: number): void
}>()

// 注入上下文（直接使用，不需要 computed 包装）
const context = inject<FolderTreeContext>(FOLDER_TREE_CONTEXT_KEY)!

// 计算展开状态
const isExpanded = computed(() => context.folderExpandState[props.folder.path] ?? true)

// 计算数量（递归）
function getImageCount(node: FolderNode): number {
  return node.images.length + node.subfolders.reduce((acc, sub) => acc + getImageCount(sub), 0)
}
</script>

<template>
  <div class="folder-node">
    <div class="folder-header" @click="context.toggleFolder(folder.path)">
      <span class="folder-icon">{{ isExpanded ? '📂' : '📁' }}</span>
      <span class="folder-name">{{ folder.name }}</span>
      <span class="folder-count">({{ getImageCount(folder) }})</span>
    </div>
    
    <div v-show="isExpanded" class="folder-content">
      <!-- 递归子文件夹 -->
      <FolderTreeNode
        v-for="subfolder in folder.subfolders"
        :key="subfolder.path"
        :folder="subfolder"
        @select="(idx) => emit('select', idx)"
        @setRef="(el, idx) => emit('setRef', el, idx)"
      />
      
      <!-- 图片列表 -->
      <div
        v-for="image in folder.images"
        :key="image.id"
        class="thumbnail-item"
        :class="{ active: context.getImageGlobalIndex(image) === context.currentIndex }"
        @click="emit('select', context.getImageGlobalIndex(image))"
      >
        <img 
          v-if="image.originalDataURL"
          :src="image.originalDataURL" 
          class="thumbnail-image"
        >
        <span class="image-name">{{ image.fileName }}</span>
        <!-- 状态指示器使用 context 获取 -->
        <span v-if="context.getStatusType(image) === 'failed'" class="translation-failed-indicator">!</span>
        <span v-else-if="context.getStatusType(image) === 'labeled'" class="labeled-indicator">✏️</span>
        <div v-if="context.getStatusType(image) === 'processing'" class="thumbnail-processing-indicator">⟳</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.folder-node {
  margin-bottom: 4px;
}

.folder-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  cursor: pointer;
  border-radius: 6px;
  transition: background-color 0.2s;
}

.folder-header:hover {
  background-color: rgba(52, 152, 219, 0.1);
}

.folder-icon {
  font-size: 14px;
}

.folder-name {
  flex: 1;
  font-size: 13px;
  font-weight: 500;
  color: #2c3e50;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.folder-count {
  font-size: 11px;
  color: #7f8c8d;
}

.folder-content {
  padding-left: 16px;
  border-left: 1px solid #e2e8f0;
  margin-left: 10px;
}

.image-name {
  display: none;
}
</style>
```

---

## ✅ 实现检查清单

- [x] 创建 `vue-frontend/src/utils/naturalSort.ts`
- [x] 更新 `vue-frontend/src/utils/index.ts` 导出 naturalSort
- [x] 修改 `vue-frontend/src/types/image.ts` 添加字段
- [x] 创建 `vue-frontend/src/types/folder.ts`
- [x] 更新 `vue-frontend/src/types/index.ts` 导出 folder
- [x] 创建 `vue-frontend/src/composables/useFolderTree.ts`
- [x] 更新 `vue-frontend/src/composables/index.ts` 导出 useFolderTree
- [x] 修改 `ImageUpload.vue` (添加文件夹选择 + 路径传递)
- [x] 创建 `FolderTreeNode.vue`
- [x] 重构 `ThumbnailSidebar.vue` (双模式渲染)
- [ ] 测试：选择包含图片的文件夹
- [ ] 测试：选择包含子文件夹的文件夹
- [ ] 测试：验证排序是否正确
- [ ] 测试：验证文件夹分组显示
- [ ] 测试：验证折叠/展开功能

---

## 🌐 浏览器兼容性

| 浏览器 | webkitdirectory 支持 |
|--------|---------------------|
| Chrome | ✅ 完全支持 |
| Edge | ✅ 完全支持 |
| Firefox | ✅ 完全支持 |
| Safari | ⚠️ 部分支持（macOS 11+） |

---

## 📅 预计工作量

| 任务 | 时间估计 |
|------|----------|
| 创建 naturalSort.ts | 5 分钟 |
| 创建 folder.ts 类型 | 3 分钟 |
| 创建 useFolderTree.ts | 10 分钟 |
| 更新索引文件 (4个) | 5 分钟 |
| 修改 ImageUpload.vue | 15 分钟 |
| 创建 FolderTreeNode.vue | 15 分钟 |
| 重构 ThumbnailSidebar.vue | 20 分钟 |
| 测试验证 | 15 分钟 |
| **总计** | **~1.5 小时** |
