<script setup lang="ts">
import './WebImportDisclaimer.global.styles.css'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
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
    overlay-class="web-import-disclaimer-overlay"
    width="90%"
    max-width="700px"
    max-height="85vh"
    border-radius="16px"
    border="2px solid var(--color-status-warning)"
    box-shadow="0 25px 80px var(--color-overlay-backdrop-strong)"
    body-padding="none"
    scroll-mode="contained"
    body-display="flex"
    body-direction="column"
    body-min-height="0"
    @update:model-value="value => { if (!value) handleCancel() }"
  >
    <div class="web-import-disclaimer-shell">
      <div class="disclaimer-header">
        <span class="warning-icon">⚠️</span>
        <h2 class="disclaimer-title">重要免责声明</h2>
      </div>

      <div class="disclaimer-content">
        <div class="disclaimer-text">
          <h3>📜 使用条款与法律声明</h3>

          <div class="section">
            <h4>1. 功能说明</h4>
            <p>
              "从网页导入"功能允许您从互联网网页中提取图片。此功能仅供<strong>技术研究与个人学习</strong>之目的提供。
            </p>
          </div>

          <div class="section">
            <h4>2. 用户责任</h4>
            <ul>
              <li>您应当确保拥有<strong>合法权利</strong>访问和下载目标内容</li>
              <li>您应当遵守目标网站的<strong>服务条款</strong>和<strong>使用协议</strong></li>
              <li>您应当尊重内容创作者的<strong>版权</strong>和<strong>知识产权</strong></li>
              <li>您<strong>不得</strong>将下载的内容用于商业目的或非法传播</li>
              <li>您<strong>不得</strong>使用本功能绕过付费内容的访问限制</li>
            </ul>
          </div>

          <div class="section">
            <h4>3. 使用限制</h4>
            <p>本功能<strong>严禁</strong>用于以下目的：</p>
            <ul>
              <li>下载、存储或传播<strong>侵权内容</strong></li>
              <li>绕过网站的<strong>付费墙</strong>或<strong>访问控制</strong></li>
              <li>进行<strong>商业用途</strong>或大规模<strong>批量爬取</strong></li>
              <li>任何违反<strong>当地法律法规</strong>的活动</li>
              <li>对目标网站造成<strong>服务器负担</strong>或<strong>恶意攻击</strong></li>
            </ul>
          </div>

          <div class="section">
            <h4>4. 免责条款</h4>
            <p>
              本软件作者及贡献者<strong>不对您使用本功能所导致的任何直接或间接后果承担责任</strong>，包括但不限于：
            </p>
            <ul>
              <li>因侵犯版权而产生的法律责任</li>
              <li>因违反服务条款而导致的账号封禁</li>
              <li>因数据丢失或损坏而造成的损失</li>
              <li>任何其他因使用本功能而产生的不利后果</li>
            </ul>
          </div>

          <div class="section warning-section">
            <h4>5. 确认声明</h4>
            <p>
              使用本功能即表示您<strong>已阅读、理解并同意</strong>上述所有条款，并承诺：
            </p>
            <ul>
              <li>仅将本功能用于<strong>合法、合规</strong>的目的</li>
              <li><strong>自行承担</strong>使用本功能所带来的一切风险和责任</li>
              <li>如因使用本功能导致任何争议，<strong>与本软件作者无关</strong></li>
            </ul>
          </div>
        </div>

        <div class="confirmation-area">
          <p class="confirmation-prompt">
            如果您已完整阅读并同意以上条款，请在下方输入框中准确输入：
          </p>
          <p class="required-text">
            <code>{{ REQUIRED_CONFIRMATION_TEXT }}</code>
          </p>
          <UiInput
            v-model="userInput"
            type="text"
            class="confirmation-input"
            :placeholder="`请输入: ${REQUIRED_CONFIRMATION_TEXT}`"
            @keyup.enter="handleConfirm"
          />
          <p v-if="userInput && !isInputCorrect" class="input-error">
            输入不正确，请完整输入「{{ REQUIRED_CONFIRMATION_TEXT }}」
          </p>
        </div>
      </div>

      <div class="disclaimer-footer">
        <UiButton variant="toolbar" class="btn-cancel" @click="handleCancel">
          我不同意，返回
        </UiButton>
        <UiButton
          variant="toolbar"
          class="btn-confirm"
          :disabled="!isInputCorrect"
          @click="handleConfirm"
        >
          ✓ 确认并继续
        </UiButton>
      </div>
    </div>
  </BaseModal>
</template>

<style scoped>
.web-import-disclaimer-shell {
  --web-import-disclaimer-header-background-end: #ffeeba;
  --web-import-disclaimer-header-divider: #f0ad4e;
  --web-import-disclaimer-title-text: #856404;
  --web-import-disclaimer-section-background: #f8f9fa;
  --web-import-disclaimer-section-accent: #6c757d;
  --web-import-disclaimer-strong-text: #c0392b;
  --web-import-disclaimer-warning-section-background: #fdf2f2;
  --web-import-disclaimer-warning-section-accent: #e74c3c;
  --web-import-disclaimer-confirmation-background-start: #e8f4fd;
  --web-import-disclaimer-confirmation-background-end: #d4eafc;
  --web-import-disclaimer-required-code-text: #2980b9;
  --web-import-disclaimer-input-focus-ring: rgba(52, 152, 219, .2);
  --web-import-disclaimer-footer-background: #f8f9fa;
  --web-import-disclaimer-cancel-border: #6c757d;
  --web-import-disclaimer-cancel-text: #6c757d;
  --web-import-disclaimer-cancel-hover-background: #6c757d;
  --web-import-disclaimer-confirm-background-end: #2ecc71;
  --web-import-disclaimer-confirm-disabled-background: #bdc3c7;
  --web-import-disclaimer-confirm-hover-background-start: #219a52;
  --web-import-disclaimer-confirm-hover-shadow: rgba(39, 174, 96, .3);
  --web-import-disclaimer-scrollbar-track: #f1f1f1;
  --web-import-disclaimer-scrollbar-thumb: #c0c0c0;
  --web-import-disclaimer-scrollbar-thumb-hover: #a0a0a0;
  --web-import-disclaimer-dark-header-background-start: #3d3a1d;
  --web-import-disclaimer-dark-header-background-end: #4a4520;
  --web-import-disclaimer-dark-title-text: #ffc107;
  --web-import-disclaimer-dark-body-text: #e0e0e0;
  --web-import-disclaimer-dark-section-background: #252540;
  --web-import-disclaimer-dark-warning-section-background: #3d2525;
  --web-import-disclaimer-dark-confirmation-background-start: #1a2a3a;
  --web-import-disclaimer-dark-confirmation-background-end: #1d3040;
  --web-import-disclaimer-dark-confirmation-border: #2980b9;
  --web-import-disclaimer-dark-code-text: #5dade2;
  --web-import-disclaimer-dark-input-border: #404060;
  --web-import-disclaimer-dark-footer-background: #16162a;
  --web-import-disclaimer-dark-cancel-text: #aaa;
  --web-import-disclaimer-dark-cancel-border: #555;
  --web-import-disclaimer-dark-cancel-hover-background: #555;

  display: flex;
  flex: 1;
  flex-direction: column;
  min-height: 0;
}

.disclaimer-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px 24px;
  background: linear-gradient(135deg, var(--color-status-warning-surface), var(--web-import-disclaimer-header-background-end));
  border-bottom: 2px solid var(--web-import-disclaimer-header-divider);
  border-radius: 14px 14px 0 0;
}

.warning-icon {
  font-size: 32px;
}

.disclaimer-title {
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: var(--web-import-disclaimer-title-text);
}

.disclaimer-content {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
}

.disclaimer-text {
  color: var(--color-text-default);
  line-height: 1.7;
}

.disclaimer-text h3 {
  margin: 0 0 20px;
  font-size: 18px;
  color: var(--color-text-default);
  padding-bottom: 12px;
  border-bottom: 2px solid var(--color-border-muted);
}

.section {
  margin-bottom: 20px;
  padding: 16px;
  background: var(--web-import-disclaimer-section-background);
  border-radius: 8px;
  border-left: 4px solid var(--web-import-disclaimer-section-accent);
}

.section h4 {
  margin: 0 0 10px;
  font-size: 15px;
  color: var(--color-text-default);
}

.section p {
  margin: 0 0 8px;
  font-size: 14px;
}

.section ul {
  margin: 8px 0 0;
  padding-left: 20px;
}

.section li {
  margin-bottom: 6px;
  font-size: 14px;
}

.section strong {
  color: var(--web-import-disclaimer-strong-text);
}

.warning-section {
  border-left-color: var(--web-import-disclaimer-warning-section-accent);
  background: var(--web-import-disclaimer-warning-section-background);
}

.confirmation-area {
  margin-top: 24px;
  padding: 20px;
  background: linear-gradient(135deg, var(--web-import-disclaimer-confirmation-background-start), var(--web-import-disclaimer-confirmation-background-end));
  border-radius: 12px;
  border: 2px solid var(--color-border-accent);
}

.confirmation-prompt {
  margin: 0 0 12px;
  font-size: 15px;
  color: var(--color-text-default);
  font-weight: 500;
}

.required-text {
  margin: 0 0 16px;
  text-align: center;
}

.required-text code {
  display: inline-block;
  padding: 10px 24px;
  background: var(--color-surface-base);
  color: var(--web-import-disclaimer-required-code-text);
  font-size: 18px;
  font-weight: 700;
  border-radius: 8px;
  border: 2px dashed var(--color-border-accent);
  font-family: var(--font-sans);
}

.confirmation-input {
  width: 100%;
  padding: 14px 16px;
  font-size: 16px;
  border: 2px solid var(--color-border-muted);
  border-radius: 8px;
  outline: none;
  transition: all 0.2s;
  text-align: center;
  background: var(--color-surface-base);
}

.confirmation-input:focus {
  border-color: var(--color-border-accent);
  box-shadow: 0 0 0 3px var(--web-import-disclaimer-input-focus-ring);
}

.input-error {
  margin: 10px 0 0;
  font-size: 13px;
  color: var(--color-text-danger-strong);
  text-align: center;
}

.disclaimer-footer {
  display: flex;
  gap: 12px;
  padding: 20px 24px;
  border-top: 1px solid var(--color-border-muted);
  background: var(--web-import-disclaimer-footer-background);
  border-radius: 0 0 14px 14px;
}

.btn-cancel {
  flex: 1;
  padding: 14px 20px;
  font-size: 15px;
  font-weight: 500;
  border: 2px solid var(--web-import-disclaimer-cancel-border);
  background: var(--color-surface-base);
  color: var(--web-import-disclaimer-cancel-text);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-cancel:hover {
  background: var(--web-import-disclaimer-cancel-hover-background);
  color: var(--color-text-inverse);
}

.btn-confirm {
  flex: 1;
  padding: 14px 20px;
  font-size: 15px;
  font-weight: 600;
  border: none;
  background: linear-gradient(135deg, var(--color-surface-success), var(--web-import-disclaimer-confirm-background-end));
  color: var(--color-text-inverse);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-confirm:disabled {
  background: var(--web-import-disclaimer-confirm-disabled-background);
  cursor: not-allowed;
  opacity: 0.7;
}

.btn-confirm:hover:not(:disabled) {
  background: linear-gradient(135deg, var(--web-import-disclaimer-confirm-hover-background-start), var(--color-surface-success));
  transform: translateY(-1px);
  box-shadow: 0 4px 12px var(--web-import-disclaimer-confirm-hover-shadow);
}

.disclaimer-content::-webkit-scrollbar {
  width: 8px;
}

.disclaimer-content::-webkit-scrollbar-track {
  background: var(--web-import-disclaimer-scrollbar-track);
  border-radius: 4px;
}

.disclaimer-content::-webkit-scrollbar-thumb {
  background: var(--web-import-disclaimer-scrollbar-thumb);
  border-radius: 4px;
}

.disclaimer-content::-webkit-scrollbar-thumb:hover {
  background: var(--web-import-disclaimer-scrollbar-thumb-hover);
}

@media (prefers-color-scheme: dark) {
  .disclaimer-header {
    background: linear-gradient(135deg, var(--web-import-disclaimer-dark-header-background-start), var(--web-import-disclaimer-dark-header-background-end));
  }

  .disclaimer-title {
    color: var(--web-import-disclaimer-dark-title-text);
  }

  .disclaimer-text,
  .disclaimer-text h3,
  .section h4,
  .confirmation-prompt {
    color: var(--web-import-disclaimer-dark-body-text);
  }

  .section {
    background: var(--web-import-disclaimer-dark-section-background);
  }

  .warning-section {
    background: var(--web-import-disclaimer-dark-warning-section-background);
  }

  .confirmation-area {
    background: linear-gradient(135deg, var(--web-import-disclaimer-dark-confirmation-background-start), var(--web-import-disclaimer-dark-confirmation-background-end));
    border-color: var(--web-import-disclaimer-dark-confirmation-border);
  }

  .required-text code {
    background: var(--web-import-disclaimer-dark-section-background);
    color: var(--web-import-disclaimer-dark-code-text);
  }

  .confirmation-input {
    background: var(--web-import-disclaimer-dark-section-background);
    color: var(--web-import-disclaimer-dark-body-text);
    border-color: var(--web-import-disclaimer-dark-input-border);
  }

  .disclaimer-footer {
    background: var(--web-import-disclaimer-dark-footer-background);
  }

  .btn-cancel {
    background: transparent;
    color: var(--web-import-disclaimer-dark-cancel-text);
    border-color: var(--web-import-disclaimer-dark-cancel-border);
  }

  .btn-cancel:hover {
    background: var(--web-import-disclaimer-dark-cancel-hover-background);
    color: var(--color-text-inverse);
  }
}
</style>
