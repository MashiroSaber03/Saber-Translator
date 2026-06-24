<script setup lang="ts">
import './EditExitSaveModal.global.styles.css'
import { computed } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
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
    custom-class="edit-exit-save-modal"
    overlay-class="edit-exit-save-modal-overlay"
    width="min(360px, calc(100vw - 32px))"
    header-padding="16px 16px 0"
    header-border="0"
    title-color="var(--color-text-inverse)"
    title-font-size="15px"
    title-font-weight="600"
    body-padding-value="8px 16px 0"
    footer-padding="14px 16px 16px"
    footer-border="0"
    footer-gap="8px"
    footer-justify="flex-start"
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
      <UiButton
        v-if="state === 'confirm'"
        class="exit-save-dialog-btn exit-save-dialog-btn--secondary"
        variant="secondary"
        data-testid="exit-without-save-button"
        @click="emit('exitWithoutSaving')"
      >
        直接退出
      </UiButton>
      <UiButton
        class="exit-save-dialog-btn exit-save-dialog-btn--primary"
        variant="primary"
        :data-testid="state === 'confirm' ? 'save-and-exit-button' : 'retry-save-and-exit-button'"
        @click="emit('saveAndExit')"
      >
        {{ state === 'confirm' ? '保存后退出' : '重试保存' }}
      </UiButton>
      <UiButton
        class="exit-save-dialog-btn exit-save-dialog-btn--ghost"
        variant="toolbar"
        :data-testid="state === 'confirm' ? 'cancel-exit-save-button' : 'return-to-editing-button'"
        @click="emit('cancel')"
      >
        {{ state === 'confirm' ? '取消' : '返回编辑' }}
      </UiButton>
    </template>
  </BaseModal>
</template>

<style scoped>
.exit-save-dialog-text,
.exit-save-dialog-btn,
.exit-save-dialog-progress,
.exit-save-dialog-progress-bar,
.exit-save-dialog-progress-fill,
.exit-save-dialog-progress-meta {
  --edit-exit-save-modal-border-strong: rgba(255, 255, 255, .16);
  --edit-exit-save-modal-border-muted: rgba(255, 255, 255, .24);
  --edit-exit-save-modal-shadow-raised: rgba(0, 255, 136, .18);
  --edit-exit-save-modal-surface-muted: #0f8;
  --edit-exit-save-modal-surface-subtle: #00cc6a;
  --edit-exit-save-modal-surface-hover: rgba(255, 255, 255, .08);
  --edit-exit-save-modal-surface-active: #00d4ff;
  --edit-exit-save-modal-text-primary: rgba(255, 255, 255, .82);
  --edit-exit-save-modal-text-secondary: #11212f;
  --edit-exit-save-modal-text-muted: #0f8;
}

.exit-save-dialog-text {
  margin: 0;
  color: var(--edit-exit-save-modal-text-primary);
  font-size: 13px;
  line-height: 1.6;
}

.exit-save-dialog-btn {
  min-width: 88px;
  padding: 8px 14px;
  border: 1px solid transparent;
  border-radius: 8px;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.exit-save-dialog-btn--primary {
  background: linear-gradient(135deg, var(--edit-exit-save-modal-surface-muted) 0%, var(--edit-exit-save-modal-surface-subtle) 100%);
  color: var(--edit-exit-save-modal-text-secondary);
  font-weight: 600;
}

.exit-save-dialog-btn--primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 24px var(--edit-exit-save-modal-shadow-raised);
}

.exit-save-dialog-btn--secondary,
.exit-save-dialog-btn--ghost {
  border-color: var(--edit-exit-save-modal-border-strong);
  background: var(--edit-exit-save-modal-surface-hover);
  color: var(--color-text-inverse);
}

.exit-save-dialog-btn--secondary:hover,
.exit-save-dialog-btn--ghost:hover {
  border-color: var(--edit-exit-save-modal-border-muted);
  background: var(--color-surface-overlay-light-prominent);
}

.exit-save-dialog-progress {
  margin-top: 14px;
}

.exit-save-dialog-progress-bar {
  width: 100%;
  height: 8px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--color-surface-overlay-light-strong);
}

.exit-save-dialog-progress-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--edit-exit-save-modal-surface-muted) 0%, var(--edit-exit-save-modal-surface-active) 100%);
  transition: width 0.25s ease;
}

.exit-save-dialog-progress-meta {
  margin-top: 8px;
  color: var(--edit-exit-save-modal-text-muted);
  font-size: 12px;
  font-weight: 600;
  text-align: right;
}
</style>
