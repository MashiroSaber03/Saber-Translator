<script setup lang="ts">
import { computed } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'

type ExitDialogState = 'closed' | 'confirm' | 'saving' | 'error'

const props = defineProps<{
  state: ExitDialogState
  message: string
  error: string
  progressPercent: number
  hasProgress: boolean
  current: number
  total: number
}>()

const emit = defineEmits<{
  cancel: []
  exitWithoutSaving: []
  saveAndExit: []
}>()

const isSaving = computed(() => props.state === 'saving')
const title = computed(() => {
  if (props.state === 'error') return '保存失败'
  if (props.state === 'saving') return '保存后退出'
  return '退出编辑'
})

function requestClose(): void {
  if (!isSaving.value) {
    emit('cancel')
  }
}
</script>

<template>
  <BaseModal
    :model-value="true"
    :title="title"
    size="small"
    placement="top-end"
    backdrop="strong"
    chrome-variant="inverse"
    divider-variant="none"
    width="min(360px, calc(100vw - 32px))"
    header-padding="16px 16px 0"
    body-padding-value="8px 16px 0"
    footer-padding="14px 16px 16px"
    footer-wrap="wrap"
    :show-close-button="false"
    :close-on-overlay="!isSaving"
    :close-on-esc="!isSaving"
    @update:model-value="requestClose"
    @close="requestClose"
  >
    <template v-if="state === 'confirm'">
      <p class="exit-save-dialog-text">是否进行全量保存（避免丢失编辑数据）</p>
    </template>

    <template v-else-if="state === 'saving'">
      <p class="exit-save-dialog-text">{{ message }}</p>
      <div class="exit-save-dialog-progress">
        <div
          class="exit-save-dialog-progress-bar"
          role="progressbar"
          aria-label="退出编辑保存进度"
          aria-valuemin="0"
          :aria-valuemax="total"
          :aria-valuenow="current"
        >
          <div
            class="exit-save-dialog-progress-fill"
            :style="{ width: `${progressPercent}%` }"
          ></div>
        </div>
        <div v-if="hasProgress" class="exit-save-dialog-progress-meta">
          {{ current }}/{{ total }}
        </div>
      </div>
    </template>

    <template v-else>
      <p class="exit-save-dialog-text">{{ error }}</p>
    </template>

    <template v-if="state !== 'saving'" #footer>
      <ProductActionRow
        class="exit-save-dialog-actions"
        variant="dialog"
        justify="start"
        aria-label="退出编辑保存操作"
      >
        <UiButton
          v-if="state === 'confirm'"
          variant="secondary"
          data-testid="exit-without-save-button"
          @click="emit('exitWithoutSaving')"
        >
          直接退出
        </UiButton>
        <UiButton
          variant="primary"
          :data-testid="state === 'confirm' ? 'save-and-exit-button' : 'retry-save-and-exit-button'"
          @click="emit('saveAndExit')"
        >
          {{ state === 'confirm' ? '保存后退出' : '重试保存' }}
        </UiButton>
        <UiButton
          variant="secondary"
          :data-testid="state === 'confirm' ? 'cancel-exit-save-button' : 'return-to-editing-button'"
          @click="emit('cancel')"
        >
          {{ state === 'confirm' ? '取消' : '返回编辑' }}
        </UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<style scoped>
.exit-save-dialog-text,
.exit-save-dialog-progress,
.exit-save-dialog-progress-bar,
.exit-save-dialog-progress-fill,
.exit-save-dialog-progress-meta {
  --edit-exit-save-modal-dialog-text: color-mix(in srgb, var(--color-text-inverse) 82%, transparent);
  --edit-exit-save-modal-progress-fill-start: var(--color-action-success-strong);
  --edit-exit-save-modal-progress-fill-end: var(--color-status-info);
  --edit-exit-save-modal-progress-meta-text: var(--color-action-success-strong);
}

.exit-save-dialog-text {
  margin: 0;
  color: var(--edit-exit-save-modal-dialog-text);
  font-size: 13px;
  line-height: 1.6;
}

.exit-save-dialog-progress {
  margin-top: 14px;
}

.exit-save-dialog-progress-bar {
  width: 100%;
  height: 8px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--color-overlay-inverse-strong);
}

.exit-save-dialog-progress-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--edit-exit-save-modal-progress-fill-start) 0%, var(--edit-exit-save-modal-progress-fill-end) 100%);
  transition: width 0.25s ease;
}

.exit-save-dialog-progress-meta {
  margin-top: 8px;
  color: var(--edit-exit-save-modal-progress-meta-text);
  font-size: 12px;
  font-weight: 600;
  text-align: right;
}
</style>
