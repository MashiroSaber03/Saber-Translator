<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiField from '@/components/ui/UiField.vue'
import BaseModal from '@/components/common/BaseModal.vue'
import { ref, computed } from 'vue'
import { useWebImportStore } from '@/stores/webImportStore'

const webImportStore = useWebImportStore()

const REQUIRED_CONFIRMATION_TEXT = '我已阅读并同意'
const userInput = ref('')
const isVisible = computed(() => webImportStore.disclaimerVisible)
const isInputCorrect = computed(() =>
  userInput.value.trim() === REQUIRED_CONFIRMATION_TEXT
)
const confirmationError = computed(() => (
  userInput.value && !isInputCorrect.value
    ? `输入不正确，请完整输入「${REQUIRED_CONFIRMATION_TEXT}」`
    : ''
))

function handleConfirm() {
  if (isInputCorrect.value) {
    webImportStore.acceptDisclaimer()
    userInput.value = ''
  }
}

function handleCancel() {
  webImportStore.rejectDisclaimer()
  userInput.value = ''
}

</script>

<template>
  <BaseModal
    :model-value="isVisible"
    :show-header="false"
    custom-class="web-import-disclaimer-modal"
    backdrop="strong"
    overlay-layer="popover"
    backdrop-effect="blur-sm"
    frame-variant="warning"
    width="90%"
    max-width="700px"
    max-height="85vh"
    body-padding="none"
    scroll-mode="contained"
    body-display="flex"
    body-direction="column"
    body-min-height="0"
    @update:model-value="value => { if (!value) handleCancel() }"
  >
    <div class="web-import-disclaimer">
      <div class="web-import-disclaimer__header">
        <UiIcon name="alert-triangle" class="web-import-disclaimer__warning-icon" size="32" />
        <h2 class="web-import-disclaimer__title">重要免责声明</h2>
      </div>

      <div class="web-import-disclaimer__content">
        <div class="web-import-disclaimer__text">
          <h3 class="web-import-disclaimer__terms-heading">
            <UiIcon name="file-text" />
            <span>使用条款与法律声明</span>
          </h3>

          <div class="web-import-disclaimer__section">
            <h4 class="web-import-disclaimer__section-title">1. 功能说明</h4>
            <p class="web-import-disclaimer__paragraph">
              "从网页导入"功能允许您从互联网网页中提取图片。此功能仅供<strong class="web-import-disclaimer__emphasis">技术研究与个人学习</strong>之目的提供。
            </p>
          </div>

          <div class="web-import-disclaimer__section">
            <h4 class="web-import-disclaimer__section-title">2. 用户责任</h4>
            <ul class="web-import-disclaimer__list">
              <li class="web-import-disclaimer__list-item">您应当确保拥有<strong class="web-import-disclaimer__emphasis">合法权利</strong>访问和下载目标内容</li>
              <li class="web-import-disclaimer__list-item">您应当遵守目标网站的<strong class="web-import-disclaimer__emphasis">服务条款</strong>和<strong class="web-import-disclaimer__emphasis">使用协议</strong></li>
              <li class="web-import-disclaimer__list-item">您应当尊重内容创作者的<strong class="web-import-disclaimer__emphasis">版权</strong>和<strong class="web-import-disclaimer__emphasis">知识产权</strong></li>
              <li class="web-import-disclaimer__list-item">您<strong class="web-import-disclaimer__emphasis">不得</strong>将下载的内容用于商业目的或非法传播</li>
              <li class="web-import-disclaimer__list-item">您<strong class="web-import-disclaimer__emphasis">不得</strong>使用本功能绕过付费内容的访问限制</li>
            </ul>
          </div>

          <div class="web-import-disclaimer__section">
            <h4 class="web-import-disclaimer__section-title">3. 使用限制</h4>
            <p class="web-import-disclaimer__paragraph">本功能<strong class="web-import-disclaimer__emphasis">严禁</strong>用于以下目的：</p>
            <ul class="web-import-disclaimer__list">
              <li class="web-import-disclaimer__list-item">下载、存储或传播<strong class="web-import-disclaimer__emphasis">侵权内容</strong></li>
              <li class="web-import-disclaimer__list-item">绕过网站的<strong class="web-import-disclaimer__emphasis">付费墙</strong>或<strong class="web-import-disclaimer__emphasis">访问控制</strong></li>
              <li class="web-import-disclaimer__list-item">进行<strong class="web-import-disclaimer__emphasis">商业用途</strong>或大规模<strong class="web-import-disclaimer__emphasis">批量爬取</strong></li>
              <li class="web-import-disclaimer__list-item">任何违反<strong class="web-import-disclaimer__emphasis">当地法律法规</strong>的活动</li>
              <li class="web-import-disclaimer__list-item">对目标网站造成<strong class="web-import-disclaimer__emphasis">服务器负担</strong>或<strong class="web-import-disclaimer__emphasis">恶意攻击</strong></li>
            </ul>
          </div>

          <div class="web-import-disclaimer__section">
            <h4 class="web-import-disclaimer__section-title">4. 免责条款</h4>
            <p class="web-import-disclaimer__paragraph">
              本软件作者及贡献者<strong class="web-import-disclaimer__emphasis">不对您使用本功能所导致的任何直接或间接后果承担责任</strong>，包括但不限于：
            </p>
            <ul class="web-import-disclaimer__list">
              <li class="web-import-disclaimer__list-item">因侵犯版权而产生的法律责任</li>
              <li class="web-import-disclaimer__list-item">因违反服务条款而导致的账号封禁</li>
              <li class="web-import-disclaimer__list-item">因数据丢失或损坏而造成的损失</li>
              <li class="web-import-disclaimer__list-item">任何其他因使用本功能而产生的不利后果</li>
            </ul>
          </div>

          <div class="web-import-disclaimer__section web-import-disclaimer__section--warning">
            <h4 class="web-import-disclaimer__section-title">5. 确认声明</h4>
            <p class="web-import-disclaimer__paragraph">
              使用本功能即表示您<strong class="web-import-disclaimer__emphasis">已阅读、理解并同意</strong>上述所有条款，并承诺：
            </p>
            <ul class="web-import-disclaimer__list">
              <li class="web-import-disclaimer__list-item">仅将本功能用于<strong class="web-import-disclaimer__emphasis">合法、合规</strong>的目的</li>
              <li class="web-import-disclaimer__list-item"><strong class="web-import-disclaimer__emphasis">自行承担</strong>使用本功能所带来的一切风险和责任</li>
              <li class="web-import-disclaimer__list-item">如因使用本功能导致任何争议，<strong class="web-import-disclaimer__emphasis">与本软件作者无关</strong></li>
            </ul>
          </div>
        </div>

        <div class="web-import-disclaimer__confirmation">
          <p class="web-import-disclaimer__confirmation-prompt">
            如果您已完整阅读并同意以上条款，请在下方输入框中准确输入：
          </p>
          <p class="web-import-disclaimer__required-text">
            <code class="web-import-disclaimer__required-code">{{ REQUIRED_CONFIRMATION_TEXT }}</code>
          </p>
          <UiField
            variant="dialog"
            label="确认输入"
            control-id="webImportDisclaimerConfirmation"
            :error="confirmationError"
          >
            <UiInput
              id="webImportDisclaimerConfirmation"
              v-model="userInput"
              type="text"
              size="lg"
              :placeholder="`请输入: ${REQUIRED_CONFIRMATION_TEXT}`"
              @keyup.enter="handleConfirm"
            />
          </UiField>
        </div>
      </div>

      <ProductActionRow
        class="web-import-disclaimer__footer"
        variant="dialog"
        justify="between"
        aria-label="网页导入免责声明操作"
      >
        <UiButton variant="secondary" class="web-import-disclaimer__action" @click="handleCancel">
          我不同意，返回
        </UiButton>
        <UiButton
          variant="primary"
          class="web-import-disclaimer__action"
          :disabled="!isInputCorrect"
          @click="handleConfirm"
        >
          <UiIcon name="check" />
          <span>确认并继续</span>
        </UiButton>
      </ProductActionRow>
    </div>
  </BaseModal>
</template>

<style scoped>
.web-import-disclaimer {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-height: 0;
}

.web-import-disclaimer__header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px 24px;
  background: var(--color-status-warning-surface);
  border-bottom: 2px solid var(--color-status-warning);
  border-radius: 14px 14px 0 0;
}

.web-import-disclaimer__warning-icon {
  color: var(--color-status-warning);
}

.web-import-disclaimer__title {
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: var(--color-text-heading);
}

.web-import-disclaimer__content {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
}

.web-import-disclaimer__text {
  color: var(--color-text-default);
  line-height: 1.7;
}

.web-import-disclaimer__terms-heading {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0 0 20px;
  font-size: 18px;
  color: var(--color-text-default);
  padding-bottom: 12px;
  border-bottom: 2px solid var(--color-border-muted);
}

.web-import-disclaimer__section {
  margin-bottom: 20px;
  padding: 16px;
  background: var(--color-surface-raised);
  border-radius: 8px;
  border-left: 4px solid var(--color-border-muted);
}

.web-import-disclaimer__section-title {
  margin: 0 0 10px;
  font-size: 15px;
  color: var(--color-text-default);
}

.web-import-disclaimer__paragraph {
  margin: 0 0 8px;
  font-size: 14px;
}

.web-import-disclaimer__list {
  margin: 8px 0 0;
  padding-left: 20px;
}

.web-import-disclaimer__list-item {
  margin-bottom: 6px;
  font-size: 14px;
}

.web-import-disclaimer__emphasis {
  color: var(--color-text-danger-strong);
}

.web-import-disclaimer__section--warning {
  border-left-color: var(--color-status-error);
  background: var(--color-surface-danger-soft);
}

.web-import-disclaimer__confirmation {
  margin-top: 24px;
  padding: 20px;
  background: linear-gradient(135deg, var(--color-surface-interactive-hover), var(--color-surface-muted));
  border-radius: 12px;
  border: 2px solid var(--color-border-accent);
}

.web-import-disclaimer__confirmation-prompt {
  margin: 0 0 12px;
  font-size: 15px;
  color: var(--color-text-default);
  font-weight: 500;
}

.web-import-disclaimer__required-text {
  margin: 0 0 16px;
  text-align: center;
}

.web-import-disclaimer__required-code {
  display: inline-block;
  padding: 10px 24px;
  background: var(--color-surface-base);
  color: var(--color-text-link);
  font-size: 18px;
  font-weight: 700;
  border-radius: 8px;
  border: 2px dashed var(--color-border-accent);
  font-family: var(--font-sans);
}

.web-import-disclaimer__footer {
  padding: 20px 24px;
  border-top: 1px solid var(--color-border-muted);
  background: var(--color-surface-quiet);
  border-radius: 0 0 14px 14px;
  flex-shrink: 0;
}

.web-import-disclaimer__action {
  flex: 1 1 220px;
}

.web-import-disclaimer__content::-webkit-scrollbar {
  width: 8px;
}

.web-import-disclaimer__content::-webkit-scrollbar-track {
  background: var(--color-surface-muted);
  border-radius: 4px;
}

.web-import-disclaimer__content::-webkit-scrollbar-thumb {
  background: var(--color-border-default);
  border-radius: 4px;
}

.web-import-disclaimer__content::-webkit-scrollbar-thumb:hover {
  background: var(--color-text-muted);
}
</style>
