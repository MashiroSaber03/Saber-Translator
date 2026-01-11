<script setup lang="ts">
/**
 * 网页导入功能免责声明弹窗
 * 用户必须输入指定确认文本才能使用该功能
 */
import { ref, computed } from 'vue'
import { useWebImportStore } from '@/stores/webImportStore'

const webImportStore = useWebImportStore()

// 用户需要输入的确认文本
const REQUIRED_CONFIRMATION_TEXT = '我已阅读并同意'

// 用户输入的文本
const userInput = ref('')

// 是否可见
const isVisible = computed(() => webImportStore.disclaimerVisible)

// 检查输入是否正确
const isInputCorrect = computed(() => 
  userInput.value.trim() === REQUIRED_CONFIRMATION_TEXT
)

// 提交同意
function handleConfirm() {
  if (isInputCorrect.value) {
    webImportStore.acceptDisclaimer()
    userInput.value = ''
  }
}

// 取消/拒绝
function handleCancel() {
  webImportStore.rejectDisclaimer()
  userInput.value = ''
}
</script>

<template>
  <Teleport to="body">
    <div v-if="isVisible" class="disclaimer-overlay" @click.self="handleCancel">
      <div class="disclaimer-container">
        <!-- 标题 -->
        <div class="disclaimer-header">
          <span class="warning-icon">⚠️</span>
          <h2 class="disclaimer-title">重要免责声明</h2>
        </div>

        <!-- 内容 -->
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

          <!-- 确认输入区域 -->
          <div class="confirmation-area">
            <p class="confirmation-prompt">
              如果您已完整阅读并同意以上条款，请在下方输入框中准确输入：
            </p>
            <p class="required-text">
              <code>{{ REQUIRED_CONFIRMATION_TEXT }}</code>
            </p>
            <input
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

        <!-- 底部按钮 -->
        <div class="disclaimer-footer">
          <button class="btn-cancel" @click="handleCancel">
            我不同意，返回
          </button>
          <button 
            class="btn-confirm" 
            :disabled="!isInputCorrect"
            @click="handleConfirm"
          >
            ✓ 确认并继续
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.disclaimer-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
  backdrop-filter: blur(4px);
}

.disclaimer-container {
  background: var(--bg-primary, #fff);
  border-radius: 16px;
  width: 90%;
  max-width: 700px;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 25px 80px rgba(0, 0, 0, 0.4);
  border: 2px solid #f0ad4e;
}

.disclaimer-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px 24px;
  background: linear-gradient(135deg, #fff3cd, #ffeeba);
  border-bottom: 2px solid #f0ad4e;
  border-radius: 14px 14px 0 0;
}

.warning-icon {
  font-size: 32px;
}

.disclaimer-title {
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: #856404;
}

.disclaimer-content {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
}

.disclaimer-text {
  color: var(--text-primary, #333);
  line-height: 1.7;
}

.disclaimer-text h3 {
  margin: 0 0 20px;
  font-size: 18px;
  color: var(--text-primary, #333);
  padding-bottom: 12px;
  border-bottom: 2px solid var(--border-color, #eee);
}

.section {
  margin-bottom: 20px;
  padding: 16px;
  background: var(--bg-secondary, #f8f9fa);
  border-radius: 8px;
  border-left: 4px solid #6c757d;
}

.section h4 {
  margin: 0 0 10px;
  font-size: 15px;
  color: var(--text-primary, #333);
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
  color: #c0392b;
}

.warning-section {
  border-left-color: #e74c3c;
  background: #fdf2f2;
}

.confirmation-area {
  margin-top: 24px;
  padding: 20px;
  background: linear-gradient(135deg, #e8f4fd, #d4eafc);
  border-radius: 12px;
  border: 2px solid #3498db;
}

.confirmation-prompt {
  margin: 0 0 12px;
  font-size: 15px;
  color: var(--text-primary, #333);
  font-weight: 500;
}

.required-text {
  margin: 0 0 16px;
  text-align: center;
}

.required-text code {
  display: inline-block;
  padding: 10px 24px;
  background: #fff;
  color: #2980b9;
  font-size: 18px;
  font-weight: 700;
  border-radius: 8px;
  border: 2px dashed #3498db;
  font-family: system-ui, -apple-system, sans-serif;
}

.confirmation-input {
  width: 100%;
  padding: 14px 16px;
  font-size: 16px;
  border: 2px solid var(--border-color, #ddd);
  border-radius: 8px;
  outline: none;
  transition: all 0.2s;
  text-align: center;
  background: #fff;
}

.confirmation-input:focus {
  border-color: #3498db;
  box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
}

.input-error {
  margin: 10px 0 0;
  font-size: 13px;
  color: #e74c3c;
  text-align: center;
}

.disclaimer-footer {
  display: flex;
  gap: 12px;
  padding: 20px 24px;
  border-top: 1px solid var(--border-color, #eee);
  background: var(--bg-secondary, #f8f9fa);
  border-radius: 0 0 14px 14px;
}

.btn-cancel {
  flex: 1;
  padding: 14px 20px;
  font-size: 15px;
  font-weight: 500;
  border: 2px solid #6c757d;
  background: #fff;
  color: #6c757d;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-cancel:hover {
  background: #6c757d;
  color: #fff;
}

.btn-confirm {
  flex: 1;
  padding: 14px 20px;
  font-size: 15px;
  font-weight: 600;
  border: none;
  background: linear-gradient(135deg, #27ae60, #2ecc71);
  color: #fff;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-confirm:hover:not(:disabled) {
  background: linear-gradient(135deg, #219a52, #27ae60);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
}

.btn-confirm:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
  opacity: 0.7;
}

/* 滚动条样式 */
.disclaimer-content::-webkit-scrollbar {
  width: 8px;
}

.disclaimer-content::-webkit-scrollbar-track {
  background: var(--bg-secondary, #f1f1f1);
  border-radius: 4px;
}

.disclaimer-content::-webkit-scrollbar-thumb {
  background: #c0c0c0;
  border-radius: 4px;
}

.disclaimer-content::-webkit-scrollbar-thumb:hover {
  background: #a0a0a0;
}

/* 暗色模式适配 */
@media (prefers-color-scheme: dark) {
  .disclaimer-container {
    background: #1a1a2e;
    border-color: #f0ad4e;
  }

  .disclaimer-header {
    background: linear-gradient(135deg, #3d3a1d, #4a4520);
  }

  .disclaimer-title {
    color: #ffc107;
  }

  .disclaimer-text,
  .disclaimer-text h3,
  .section h4,
  .confirmation-prompt {
    color: #e0e0e0;
  }

  .section {
    background: #252540;
  }

  .warning-section {
    background: #3d2525;
  }

  .confirmation-area {
    background: linear-gradient(135deg, #1a2a3a, #1d3040);
    border-color: #2980b9;
  }

  .required-text code {
    background: #252540;
    color: #5dade2;
  }

  .confirmation-input {
    background: #252540;
    color: #e0e0e0;
    border-color: #404060;
  }

  .disclaimer-footer {
    background: #16162a;
  }

  .btn-cancel {
    background: transparent;
    color: #aaa;
    border-color: #555;
  }

  .btn-cancel:hover {
    background: #555;
    color: #fff;
  }
}
</style>
