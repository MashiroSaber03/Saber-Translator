<template>
  <div class="page-details-panel">
    <ProductSectionHeader title="页面剧情管理" icon-name="file-text" />

    <ProductEmptyState
      v-if="pages.length === 0"
      class="page-details-panel__empty"
      icon-name="file-text"
      title="尚未生成页面剧情"
      description="生成后可逐页编辑剧情承接、对白和角色信息。"
    >
      <template #actions>
        <UiButton
          variant="primary"
          :disabled="isGenerating"
          @click="$emit('generate-details')"
        >
          <UiIcon v-if="!isGenerating" name="target" size="15" />
          <span>{{ isGenerating ? '生成中...' : '生成页面剧情' }}</span>
        </UiButton>
      </template>
    </ProductEmptyState>

    <div v-else class="page-details-panel__list">
      <ProductRecordCard v-for="page in pages" :key="page.page_number" class="page-details-panel__page-card">
        <div class="page-details-panel__page-header">
          <h4 class="page-details-panel__page-title">页面 {{ page.page_number }}</h4>
          <ProductChipList
            class="page-details-panel__status-chips"
            :aria-label="`页面 ${page.page_number} 状态`"
            :items="getStatusChipItems(page.status)"
          />
        </div>

        <div class="page-details-panel__fields">
          <UiField
            class="page-details-panel__field"
            variant="settings"
            label="上一页剧情承接"
            :control-id="`continuation-page-${page.page_number}-continuity`"
          >
            <UiTextarea
              :id="`continuation-page-${page.page_number}-continuity`"
              :model-value="page.continuity_text"
              rows="3"
              size="sm"
              variant="panel"
              class="page-details-panel__field-input"
              @update:model-value="handleStoryInput(page.page_number, 'continuity_text', $event)"
            />
          </UiField>

          <UiField
            class="page-details-panel__field"
            variant="settings"
            label="本页剧情"
            :control-id="`continuation-page-${page.page_number}-story`"
          >
            <UiTextarea
              :id="`continuation-page-${page.page_number}-story`"
              :model-value="page.story_text"
              rows="4"
              size="sm"
              variant="panel"
              class="page-details-panel__field-input"
              @update:model-value="handleStoryInput(page.page_number, 'story_text', $event)"
            />
          </UiField>

          <UiField
            class="page-details-panel__field"
            variant="settings"
            label="关键对白"
            :control-id="`continuation-page-${page.page_number}-dialogue`"
          >
            <UiTextarea
              :id="`continuation-page-${page.page_number}-dialogue`"
              :model-value="page.dialogue_text"
              rows="3"
              size="sm"
              variant="panel"
              class="page-details-panel__field-input"
              @update:model-value="handleStoryInput(page.page_number, 'dialogue_text', $event)"
            />
          </UiField>

          <UiField
            class="page-details-panel__field"
            variant="settings"
            label="角色（逗号分隔）"
            :control-id="`continuation-page-${page.page_number}-characters`"
          >
            <UiInput
              :id="`continuation-page-${page.page_number}-characters`"
              :model-value="page.characters.join(', ')"
              type="text"
              size="sm"
              class="page-details-panel__field-input"
              @update:model-value="handleCharactersInput(page.page_number, $event)"
            />
          </UiField>
        </div>
      </ProductRecordCard>

      <ProductActionRow class="page-details-panel__actions" aria-label="页面剧情操作">
        <UiButton variant="secondary" :disabled="isSaving" @click="$emit('save-changes')">
          <UiIcon v-if="!isSaving" name="save" size="15" />
          <span>{{ isSaving ? '保存中...' : '保存修改' }}</span>
        </UiButton>
      </ProductActionRow>
    </div>
  </div>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiField from '@/components/ui/UiField.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import type { PageContent } from '@/api/continuation'
import type { PageStoryField, PageStoryValue } from './pageStoryTypes'

defineProps<{
  pages: PageContent[]
  isGenerating: boolean
  isSaving?: boolean
}>()

const emit = defineEmits<{
  'generate-details': []
  'save-changes': []
  'story-change': [pageNumber: number, field: PageStoryField, value: PageStoryValue]
}>()

function parseCharacters(value: string): string[] {
  return value.split(',').map(s => s.trim()).filter(Boolean)
}

function handleStoryInput(pageNumber: number, field: Exclude<PageStoryField, 'characters'>, value: string) {
  emit('story-change', pageNumber, field, value)
}

function handleCharactersInput(pageNumber: number, value: string | number | boolean) {
  emit('story-change', pageNumber, 'characters', parseCharacters(String(value)))
}

function getStatusText(status: string): string {
  const map: Record<string, string> = {
    'pending': '待处理',
    'generating': '生成中',
    'generated': '已生成',
    'stale': '需重新生成',
    'failed': '失败'
  }
  return map[status] || status
}

function getStatusTone(status: string): ProductChipItem['tone'] {
  const map: Record<string, ProductChipItem['tone']> = {
    pending: 'warning',
    generating: 'primary',
    generated: 'success',
    stale: 'warning',
    failed: 'danger',
  }
  return map[status] || 'neutral'
}

function getStatusChipItems(status: string): ProductChipItem[] {
  return [
    {
      id: status,
      label: getStatusText(status),
      tone: getStatusTone(status),
    },
  ]
}
</script>

<style scoped>
.page-details-panel {
  min-width: 0;
}

.page-details-panel__empty {
  --product-empty-state-min-height: 260px;

  padding-block: 40px;
}

.page-details-panel__list {
  display: grid;
  gap: 16px;
}

.page-details-panel__page-card {
  --product-record-card-background: var(--color-surface-subtle);
  --product-record-card-border: var(--color-border-muted);
  --product-record-card-radius: 12px;
  --product-record-card-padding: 16px;
  --product-record-card-gap: 12px;
  --product-record-card-shadow-hover: none;
}

.page-details-panel__page-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px 12px;
  min-width: 0;
  margin-bottom: 16px;
}

.page-details-panel__page-title {
  margin: 0;
  font-size: 16px;
}

.page-details-panel__status-chips {
  flex: 0 0 auto;
}

.page-details-panel__fields {
  display: grid;
  gap: 12px;
}

.page-details-panel__field {
  margin-bottom: 0;
}

.page-details-panel__field-input {
  width: 100%;
}

.page-details-panel__actions {
  margin-top: 16px;
  justify-content: center;
}

</style>
