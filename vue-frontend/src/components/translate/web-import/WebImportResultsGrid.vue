<script setup lang="ts">
import { computed } from 'vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductSelectableImageGrid from '@/components/product/ProductSelectableImageGrid.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import type { ExtractResult, WebImportStatus } from '@/types/webImport'

const props = defineProps<{
  downloadProgress: { current: number; total: number }
  engineDisplayName: string
  error: string | null
  extractResult: ExtractResult | null
  isAllSelected: boolean
  hasMorePages?: boolean
  isLoadingMorePages?: boolean
  selectedCount: number
  selectedPages: Set<number>
  status: WebImportStatus
}>()

const emit = defineEmits<{
  (event: 'toggleAll'): void
  (event: 'togglePage', pageNum: number): void
  (event: 'loadMore'): void
}>()

const imageItems = computed(() => {
  return (
    props.extractResult?.pages.map(page => ({
      id: page.pageNumber,
      src: page.imageUrl,
      alt: `第${page.pageNumber}页`,
      label: `第 ${page.pageNumber} 页`,
      selected: props.selectedPages.has(page.pageNumber),
    })) ?? []
  )
})

const resultMetadataChips = computed<ProductChipItem[]>(() => {
  if (!props.extractResult) return []

  const items: ProductChipItem[] = [
    {
      id: 'page-count',
      label: `共 ${props.extractResult.totalPages} 张`,
      tone: 'neutral',
    },
  ]

  if (props.engineDisplayName) {
    items.push({
      id: 'engine',
      label: `引擎: ${props.engineDisplayName}`,
      tone: 'neutral',
    })
  }

  return items
})

function handleToggleImage(id: string | number): void {
  if (typeof id === 'number') {
    emit('togglePage', id)
  }
}
</script>

<template>
  <div class="web-import-results-grid">
    <ProductStatusBanner v-if="error" tone="danger" aria-live="assertive">
      {{ error }}
    </ProductStatusBanner>

    <div v-if="extractResult" class="web-import-results-grid__section">
      <ProductSectionHeader title="提取结果" icon-name="book-open" size="sm">
        <template #actions>
          <ProductChipList
            class="web-import-results-grid__details"
            aria-label="网页导入结果元信息"
            :items="resultMetadataChips"
          />
        </template>
      </ProductSectionHeader>

      <ProductActionRow
        class="web-import-results-grid__selection-row"
        aria-label="网页导入结果选择"
        justify="start"
      >
        <UiCheckbox :model-value="isAllSelected" label="全选" @change="emit('toggleAll')" />
        <span class="web-import-results-grid__selected-count">已选: {{ selectedCount }} 张</span>
      </ProductActionRow>

      <ProductSelectableImageGrid
        :items="imageItems"
        aria-label="网页导入图片选择"
        @toggle="handleToggleImage"
      />
      <ProductActionRow
        v-if="hasMorePages"
        class="web-import-results-grid__load-more"
        aria-label="加载更多网页导入候选"
        justify="center"
      >
        <UiButton
          variant="secondary"
          size="sm"
          :disabled="isLoadingMorePages"
          :loading="isLoadingMorePages"
          @click="emit('loadMore')"
        >
          {{ isLoadingMorePages ? '加载中...' : '加载更多' }}
        </UiButton>
      </ProductActionRow>
    </div>

    <div v-if="status === 'downloading'" class="web-import-results-grid__progress-section">
      <UiProgressBar
        :value="downloadProgress.current"
        :max="downloadProgress.total"
        label="网页导入下载进度"
      >
        下载进度: {{ downloadProgress.current }}/{{ downloadProgress.total }}
      </UiProgressBar>
    </div>
  </div>
</template>

<style scoped>
.web-import-results-grid__section {
  margin-bottom: 16px;
}

.web-import-results-grid__details {
  flex: 0 1 auto;
  min-width: 0;
}

.web-import-results-grid__selection-row {
  margin-bottom: 12px;
}

.web-import-results-grid__selected-count {
  color: var(--color-text-supporting);
  font-size: 13px;
}

.web-import-results-grid__load-more {
  margin-top: 12px;
}

.web-import-results-grid__progress-section {
  margin-bottom: 16px;
}
</style>
