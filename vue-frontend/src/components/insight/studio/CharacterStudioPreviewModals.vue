<script setup lang="ts">
import { computed } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChoiceCardGrid, { type ProductChoiceCardItem } from '@/components/product/ProductChoiceCardGrid.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'

interface StudioGreetingOption {
  greeting_id: string
  label: string
  content: string
}

const props = defineProps<{
  greetingOpen: boolean
  promptOpen: boolean
  imageOpen: boolean
  displayGreetings: StudioGreetingOption[]
  selectedGreetingId: string
  chatMutating: boolean
  chatStreaming: boolean
  chatPromptLoading: boolean
  promptPreview: string
  promptPreviewError: string
  imageTitle: string
  imageSrc: string
}>()

const emit = defineEmits<{
  (e: 'update:greetingOpen', value: boolean): void
  (e: 'update:promptOpen', value: boolean): void
  (e: 'update:imageOpen', value: boolean): void
  (e: 'update:selectedGreetingId', value: string): void
  (e: 'confirm-greeting-selection'): void
  (e: 'copy-prompt-preview'): void
}>()

const greetingModel = computed({
  get: () => props.greetingOpen,
  set: value => emit('update:greetingOpen', value),
})

const promptModel = computed({
  get: () => props.promptOpen,
  set: value => emit('update:promptOpen', value),
})

const imageModel = computed({
  get: () => props.imageOpen,
  set: value => emit('update:imageOpen', value),
})

const greetingItems = computed<ProductChoiceCardItem[]>(() => props.displayGreetings.map(item => ({
  id: item.greeting_id,
  label: item.label,
  description: item.content,
})))
</script>

<template>
  <BaseModal
    v-model="greetingModel"
    title="重选开场白"
    size="large"
    custom-class="studio-chat-modal"
  >
    <div class="character-studio-preview-modals__copy">
      <p class="character-studio-preview-modals__copy-text">选择一条开场白后，将归档当前会话，并以该开场白重新开启一轮新对话。</p>
    </div>
    <ProductStatusBanner
      v-if="displayGreetings.length === 0"
      icon-name="message"
      role="note"
      tone="neutral"
      title="暂无可用开场白"
    >
      当前还没有可用开场白。
    </ProductStatusBanner>
    <ProductChoiceCardGrid
      v-else
      class="character-studio-preview-modals__greeting-grid"
      accessibility-label="选择开场白"
      :items="greetingItems"
      :model-value="selectedGreetingId"
      variant="compact"
      @update:model-value="$emit('update:selectedGreetingId', $event)"
    />
    <template #footer>
      <ProductActionRow
        appearance="accent"
        class="character-studio-preview-modals__actions"
        aria-label="开场白选择操作"
        variant="dialog"
      >
        <UiButton variant="secondary" @click="greetingModel = false">取消</UiButton>
        <UiButton
          variant="primary"
          :disabled="!selectedGreetingId || chatMutating || chatStreaming"
          @click="$emit('confirm-greeting-selection')"
        >
          确认并重新开场
        </UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>

  <BaseModal
    v-model="promptModel"
    title="本轮提示词预览"
    size="large"
    custom-class="studio-chat-modal"
  >
    <ProductStatusBanner
      v-if="chatPromptLoading"
      aria-live="polite"
      icon-name="loading"
      role="status"
      title="提示词加载中..."
      tone="info"
    >
      正在生成本轮提示词预览。
    </ProductStatusBanner>
    <ProductStatusBanner
      v-else-if="promptPreviewError"
      role="alert"
      tone="warning"
      title="提示词预览不可用"
    >
      {{ promptPreviewError }}
    </ProductStatusBanner>
    <div v-else-if="promptPreview.trim()" class="character-studio-preview-modals__prompt-body">
      <ProductActionRow appearance="accent" class="character-studio-preview-modals__prompt-tools" aria-label="提示词预览操作">
        <UiButton variant="secondary" @click="$emit('copy-prompt-preview')" size="sm">
          复制内容
        </UiButton>
      </ProductActionRow>
      <pre class="character-studio-preview-modals__prompt-preview">{{ promptPreview }}</pre>
    </div>
    <ProductStatusBanner
      v-else
      data-testid="prompt-preview-empty"
      icon-name="message"
      role="note"
      tone="neutral"
      title="暂无提示词预览"
    >
      请先发送至少一条消息后再查看本轮提示词。
    </ProductStatusBanner>
  </BaseModal>

  <BaseModal
    v-model="imageModel"
    :title="imageTitle"
    size="large"
    custom-class="studio-chat-modal studio-image-modal"
  >
    <div v-if="imageSrc" class="character-studio-preview-modals__image-preview">
      <img class="character-studio-preview-modals__image" :src="imageSrc" :alt="imageTitle">
    </div>
  </BaseModal>
</template>

<style scoped>
.character-studio-preview-modals__copy-text {
  color: var(--color-text-supporting);
  font-size: 13px;
  line-height: 1.7;
}

.character-studio-preview-modals__greeting-grid {
  --product-choice-card-grid-gap: 12px;
  --product-choice-card-grid-columns: repeat(2, minmax(0, 1fr));
  --product-choice-card-grid-item-align-items: stretch;
  --product-choice-card-grid-item-justify-content: flex-start;
  --product-choice-card-grid-item-min-height: 0;
  --product-choice-card-grid-item-text-align: left;
  --product-choice-card-grid-item-background: color-mix(in srgb, var(--color-surface-page) 84%, transparent);
  --product-choice-card-grid-item-background-selected: color-mix(in srgb, var(--color-text-link-strong) 10%, transparent);
  --product-choice-card-grid-item-border: color-mix(in srgb, var(--color-text-heading) 8%, transparent);
  --product-choice-card-grid-item-border-width: 1px;
  --product-choice-card-grid-item-border-selected: color-mix(in srgb, var(--color-text-link-strong) 28%, transparent);
  --product-choice-card-grid-item-radius: 18px;
  --product-choice-card-grid-item-padding: 16px;
  --product-choice-card-grid-item-shadow-hover: 0 0 0 2px color-mix(in srgb, var(--color-text-link-strong) 16%, transparent);
  --product-choice-card-grid-item-shadow-selected: inset 0 0 0 1px color-mix(in srgb, var(--color-text-link-strong) 16%, transparent);
  --product-choice-card-grid-heading-width: 100%;
  --product-choice-card-grid-heading-justify-content: space-between;
  --product-choice-card-grid-label-padding: 4px 9px;
  --product-choice-card-grid-label-radius: 999px;
  --product-choice-card-grid-label-background: color-mix(in srgb, var(--color-text-link-strong) 10%, transparent);
  --product-choice-card-grid-label-color: var(--color-text-link-strong);
  --product-choice-card-grid-label-font-size: 11px;
  --product-choice-card-grid-label-font-weight: 400;
  --product-choice-card-grid-description-margin-top: 4px;
  --product-choice-card-grid-description-color: var(--color-text-heading);
  --product-choice-card-grid-description-font-size: 13px;
  --product-choice-card-grid-description-line-height: 1.7;
  --product-choice-card-grid-description-white-space: pre-wrap;

  margin-top: 14px;
}

.character-studio-preview-modals__prompt-preview {
  max-height: 420px;
  margin: 10px 0 0;
  padding: 14px;
  overflow: auto;
  border: 1px solid color-mix(in srgb, var(--color-text-heading) 8%, transparent);
  border-radius: 16px;
  background: color-mix(in srgb, var(--color-surface-page) 92%, transparent);
  color: var(--color-text-heading);
  font-size: 12px;
  line-height: 1.7;
  white-space: pre-wrap;
}

.character-studio-preview-modals__prompt-tools {
  justify-content: flex-end;
}

.character-studio-preview-modals__image-preview {
  display: flex;
  justify-content: center;
}

.character-studio-preview-modals__image {
  max-width: 100%;
  max-height: 70vh;
  border-radius: 16px;
  object-fit: contain;
}
</style>
