<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted } from 'vue'
import ProductBreadcrumbTrail from '@/components/product/ProductBreadcrumbTrail.vue'
import ProductFolderCard from '@/components/product/ProductFolderCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import type { ProductThumbnailGridItem } from '@/components/product/ProductThumbnailGrid.vue'
import VirtualThumbnailList from '@/components/virtual/VirtualThumbnailList.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import { useImageStore } from '@/stores/imageStore'
import { useFolderTree } from '@/composables/useFolderTree'
import { useThumbnailSelection } from '@/composables/useThumbnailSelection'
import type { ImageData } from '@/types/image'

const props = defineProps<{
  isVisible?: boolean
}>()

const emit = defineEmits<{
  (e: 'select', index: number): void
}>()

const imageStore = useImageStore()

const sidebarRef = ref<HTMLElement | null>(null)
const containerRef = ref<HTMLElement | null>(null)

const images = computed(() => imageStore.images)
const currentIndex = computed(() => imageStore.currentImageIndex)
const hasImages = computed(() => imageStore.hasImages)

const {
  useTreeMode,
  breadcrumbs,
  currentSubfolders,
  currentImages,
  currentFolderPath,
  enterFolder,
  goUp,
  navigateTo,
  getFolderImageCount,
  folderTree,
  resetToRoot
} = useFolderTree(images)

const {
  getImageGlobalIndex,
  getStatusType,
  isTranslated,
  getThumbnailTitle,
} = useThumbnailSelection(images)

const flatThumbnailItems = computed<ProductThumbnailGridItem[]>(() => {
  return images.value.map((image, index) => buildThumbnailItem(image, index))
})

const currentFolderThumbnailItems = computed<ProductThumbnailGridItem[]>(() => {
  return currentImages.value.map((image) => buildThumbnailItem(image, getImageGlobalIndex(image)))
})

watch(() => images.value.length, (newLen, previousLen) => {
  if (newLen === 0 || (previousLen === 0 && newLen > 0)) {
    resetToRoot()
  }
})

function handleClick(index: number) {
  emit('select', index)
}

function handleThumbnailSelect(id: string | number) {
  if (typeof id !== 'number') return
  handleClick(id)
}

function handleFolderClick(folderPath: string) {
  enterFolder(folderPath)
}

function handleBreadcrumbClick(path: string) {
  navigateTo(path)
}

function scrollToActiveThumbnail() {
  nextTick(() => {
    // Folder mode scrolls the inner folder list; flat mode scrolls the sidebar.
    const scrollContainer = (useTreeMode.value && containerRef.value)
      ? containerRef.value
      : sidebarRef.value
    const activeThumb = scrollContainer?.querySelector<HTMLElement>(
      `[data-product-thumbnail-id="${currentIndex.value}"]`
    )

    if (activeThumb && scrollContainer) {
      const thumbRect = activeThumb.getBoundingClientRect()
      const containerRect = scrollContainer.getBoundingClientRect()

      const thumbCenter = thumbRect.top + thumbRect.height / 2
      const containerCenter = containerRect.top + containerRect.height / 2
      const scrollOffset = thumbCenter - containerCenter

      scrollContainer.scrollTo({
        top: scrollContainer.scrollTop + scrollOffset,
        behavior: 'smooth'
      })
    }
  })
}

function buildThumbnailItem(image: ImageData, index: number): ProductThumbnailGridItem {
  const statusType = getStatusType(image)
  return {
    id: index,
    src: image.thumbnailSourceUrl,
    alt: image.fileName,
    label: String(index + 1),
    selected: index === currentIndex.value,
    marked: isTranslated(image),
    cornerLabel: statusType === 'failed'
      ? '!'
      : statusType === 'labeled'
        ? '标'
        : statusType === 'processing'
          ? '处理中'
          : undefined,
    fallbackLabel: String(index + 1),
    ariaLabel: `选择图片 ${index + 1}: ${image.fileName}`,
    disabledTitle: getThumbnailTitle(image),
  }
}

watch(currentIndex, () => {
  scrollToActiveThumbnail()
})

watch(() => props.isVisible, (newVisible, previousVisible) => {
  if (newVisible && !previousVisible) {
    scrollToActiveThumbnail()
  }
})

onMounted(() => {
  if (hasImages.value) {
    scrollToActiveThumbnail()
  }
})
</script>

<template>
  <aside ref="sidebarRef" class="thumbnail-sidebar">
    <div class="thumbnail-sidebar__card">
      <h2 class="thumbnail-sidebar__title">图片概览</h2>

      <template v-if="hasImages && useTreeMode && folderTree">
        <ProductBreadcrumbTrail
          class="thumbnail-sidebar__breadcrumb"
          :items="breadcrumbs"
          @select="handleBreadcrumbClick"
        />

        <UiButton
          v-if="currentFolderPath"
          variant="secondary"
          tone="primary"
          size="sm"
          block
          class="thumbnail-sidebar__back-button"
          @click="goUp"
        >
          <UiIcon class="thumbnail-sidebar__back-icon" name="chevron-right" size="14" />
          <span>返回上级</span>
        </UiButton>

        <div ref="containerRef" class="thumbnail-sidebar__folder-content-list">
          <ProductFolderCard
            v-for="subfolder in currentSubfolders"
            :key="subfolder.path"
            class="thumbnail-sidebar__folder-card"
            :count="getFolderImageCount(subfolder)"
            :count-id="subfolder.path"
            :folder-name="subfolder.name"
            :aria-label="`打开文件夹 ${subfolder.name}`"
            @select="handleFolderClick(subfolder.path)"
          />

          <VirtualThumbnailList
            v-if="currentFolderThumbnailItems.length > 0"
            class="thumbnail-sidebar__grid"
            aria-label="图片缩略图导航"
            :active-id="currentIndex"
            :items="currentFolderThumbnailItems"
            @select="handleThumbnailSelect"
          />

          <ProductStatusBanner
            v-if="currentSubfolders.length === 0 && currentImages.length === 0"
            tone="neutral"
            icon-name="folder-open"
            title="此文件夹为空"
          >
            返回上级文件夹继续选择图片。
          </ProductStatusBanner>
        </div>
      </template>

      <div
        v-else-if="hasImages"
        ref="containerRef"
        class="thumbnail-sidebar__list"
      >
        <VirtualThumbnailList
          class="thumbnail-sidebar__grid"
          aria-label="图片缩略图导航"
          :active-id="currentIndex"
          :items="flatThumbnailItems"
          @select="handleThumbnailSelect"
        />
      </div>

      <ProductStatusBanner
        v-else
        tone="neutral"
        icon-name="image"
        title="暂无图片"
      >
        上传图片后会在这里显示缩略图。
      </ProductStatusBanner>
    </div>
  </aside>
</template>

<style scoped>
.thumbnail-sidebar {
  --thumbnail-sidebar-scrollbar-thumb: var(--color-border-muted);
  --thumbnail-sidebar-scrollbar-track: var(--color-surface-quiet);
  --thumbnail-sidebar-title-divider: var(--color-border-muted);

  width: 100%;
  height: 100%;
  overflow-y: auto;
  padding: 20px;
  margin-left: 0;
  order: 1;
  scrollbar-width: thin;
  scrollbar-color: var(--thumbnail-sidebar-scrollbar-thumb) var(--thumbnail-sidebar-scrollbar-track);
}

.thumbnail-sidebar::-webkit-scrollbar {
  width: 8px;
}

.thumbnail-sidebar::-webkit-scrollbar-track {
  background: var(--color-surface-quiet);
  border-radius: 8px;
}

.thumbnail-sidebar::-webkit-scrollbar-thumb {
  background-color: var(--thumbnail-sidebar-scrollbar-thumb);
  border-radius: 8px;
  border: 2px solid var(--thumbnail-sidebar-scrollbar-track);
}

.thumbnail-sidebar__card {
  display: flex;
  flex-direction: column;
  min-height: 0;
  height: 100%;
  background-color: var(--color-surface-card);
  border-radius: 12px;
  box-shadow: 0 4px 12px var(--shadow-soft);
  padding: 25px;
  transition: box-shadow 0.2s;
}

.thumbnail-sidebar__card:hover {
  box-shadow: 0 6px 16px var(--shadow-medium);
}

.thumbnail-sidebar__title {
  border-bottom: 2px solid var(--thumbnail-sidebar-title-divider);
  padding-bottom: 12px;
  margin-bottom: 15px;
  color: var(--color-text-heading);
  font-size: 1.4em;
  text-align: center;
}

.thumbnail-sidebar__breadcrumb {
  margin-bottom: 10px;
}

.thumbnail-sidebar__back-button {
  justify-content: flex-start;
  margin-bottom: 12px;
}

.thumbnail-sidebar__back-icon {
  transform: rotate(180deg);
}

.thumbnail-sidebar__folder-content-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
}

.thumbnail-sidebar__folder-card {
  --product-record-card-padding: 10px 12px;
  --product-record-card-radius: 8px;
  --product-record-card-gap: 8px;
}

.thumbnail-sidebar__list {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
  padding: 0;
  margin: 0;
}

.thumbnail-sidebar__grid {
  --product-thumbnail-grid-min-size: 100%;
  --product-thumbnail-grid-aspect-ratio: 3 / 4;

  flex: 1 1 auto;
  min-height: 0;
}

@media (--breakpoint-md-down) {
  .thumbnail-sidebar {
    order: 3;
    width: 100%;
    height: auto;
    max-height: none;
    overflow: visible;
    padding: 0;
  }

  .thumbnail-sidebar__card {
    padding: 16px;
  }

  .thumbnail-sidebar__title {
    font-size: 1.2em;
  }

  .thumbnail-sidebar__grid {
    --product-thumbnail-grid-min-size: min(100%, 290px);
  }
}
</style>
