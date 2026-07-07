<script setup lang="ts">
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import { ref, onMounted } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import {
  dismissFirstTimeGuide,
  shouldShowFirstTimeGuide,
} from '@/components/translate/firstTimeGuideState'

const emit = defineEmits<{
  (e: 'openSettings'): void
}>()

const showGuide = ref(false)
const dontShowAgain = ref(false)

onMounted(() => {
  if (shouldShowFirstTimeGuide()) {
    showGuide.value = true
  }
})

function closeGuide() {
  if (dontShowAgain.value) {
    dismissFirstTimeGuide()
  }
  showGuide.value = false
}

function openSettingsAndClose() {
  dismissFirstTimeGuide()
  showGuide.value = false
  emit('openSettings')
}
</script>

<template>
  <BaseModal
    :model-value="showGuide"
    title="欢迎使用 Saber-Translator"
    @close="closeGuide"
  >
    <div class="guide-content">
      <UiIcon name="sparkles" class="guide-icon" size="44" />

      <div class="guide-message">
        <p class="guide-title">首次使用提醒</p>
        <p class="guide-text">
          在开始翻译之前，请先配置翻译服务。
        </p>
        <p class="guide-text">
          点击右上角的 <span class="highlight">设置</span> 按钮，配置以下内容：
        </p>
        <ul class="guide-list">
          <li>选择 OCR 引擎（文字识别）</li>
          <li>配置翻译服务商和 API Key</li>
          <li>（可选）配置高质量翻译和 AI 校对</li>
        </ul>
      </div>

      <ProductActionRow
        class="guide-actions"
        variant="dialog"
        justify="center"
        aria-label="首次使用设置操作"
      >
        <UiButton
          variant="primary"
          @click="openSettingsAndClose"
        >
          <UiIcon name="settings" />
          <span>立即配置</span>
        </UiButton>
        <UiButton
          variant="secondary"
          @click="closeGuide"
        >
          稍后配置
        </UiButton>
      </ProductActionRow>

      <div class="guide-footer">
        <UiCheckbox
          v-model="dontShowAgain"
          label="不再显示此提醒"
        />
      </div>
    </div>
  </BaseModal>
</template>

<style scoped>
.guide-content {
  text-align: center;
  padding: 16px 0;
}

.guide-icon {
  margin-bottom: 16px;
  color: var(--color-action-primary);
}

.guide-message {
  margin-bottom: 24px;
}

.guide-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--color-text-default);
  margin-bottom: 12px;
}

.guide-text {
  font-size: 14px;
  color: var(--color-text-supporting);
  margin-bottom: 8px;
  line-height: 1.6;
}

.highlight {
  color: var(--color-action-primary);
  font-weight: 500;
}

.guide-list {
  text-align: left;
  margin: 16px auto;
  max-width: 280px;
  padding-left: 20px;
}

.guide-list li {
  font-size: 14px;
  color: var(--color-text-supporting);
  margin-bottom: 8px;
  line-height: 1.5;
}

.guide-actions {
  margin-bottom: 16px;
}

.guide-footer {
  padding-top: 12px;
  border-top: 1px solid var(--color-border-muted);
}
</style>
