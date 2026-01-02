<script setup lang="ts">
/**
 * 网页导入设置组件
 */
import { ref, computed } from 'vue'
import { useWebImportStore } from '@/stores/webImportStore'
import { testFirecrawlConnection, testAgentConnection } from '@/api/webImport'
import { WEB_IMPORT_AGENT_PROVIDERS } from '@/constants'

const store = useWebImportStore()

// 当前选项卡
const activeTab = ref<'basic' | 'preprocess' | 'advanced'>('basic')

// 测试状态
const testingFirecrawl = ref(false)
const testingAgent = ref(false)

// 显示 API Key
const showFirecrawlKey = ref(false)
const showAgentKey = ref(false)

// 计算属性
const settings = computed(() => store.settings)

// 测试 Firecrawl 连接
async function handleTestFirecrawl() {
  if (!settings.value.firecrawl.apiKey) {
    alert('请输入 Firecrawl API Key')
    return
  }

  testingFirecrawl.value = true
  try {
    const result = await testFirecrawlConnection(settings.value.firecrawl.apiKey)
    if (result.success) {
      alert('✅ Firecrawl 连接成功')
    } else {
      alert(`❌ 连接失败: ${result.error}`)
    }
  } catch (e) {
    alert(`❌ 连接失败: ${e instanceof Error ? e.message : '未知错误'}`)
  } finally {
    testingFirecrawl.value = false
  }
}

// 测试 Agent 连接
async function handleTestAgent() {
  if (!settings.value.agent.apiKey) {
    alert('请输入 AI Agent API Key')
    return
  }

  testingAgent.value = true
  try {
    const result = await testAgentConnection(
      settings.value.agent.provider,
      settings.value.agent.apiKey,
      settings.value.agent.customBaseUrl,
      settings.value.agent.modelName
    )
    if (result.success) {
      alert('✅ AI Agent 连接成功')
    } else {
      alert(`❌ 连接失败: ${result.error}`)
    }
  } catch (e) {
    alert(`❌ 连接失败: ${e instanceof Error ? e.message : '未知错误'}`)
  } finally {
    testingAgent.value = false
  }
}

// 重置提示词
function handleResetPrompt() {
  if (confirm('确定要重置为默认提示词吗？')) {
    store.resetExtractionPrompt()
  }
}

// 是否显示自定义 URL
const showCustomUrl = computed(() => settings.value.agent.provider === 'custom_openai')
</script>

<template>
  <div class="web-import-settings">
    <!-- 选项卡 -->
    <div class="tabs">
      <button
        class="tab"
        :class="{ active: activeTab === 'basic' }"
        @click="activeTab = 'basic'"
      >
        基本设置
      </button>
      <button
        class="tab"
        :class="{ active: activeTab === 'preprocess' }"
        @click="activeTab = 'preprocess'"
      >
        图片预处理
      </button>
      <button
        class="tab"
        :class="{ active: activeTab === 'advanced' }"
        @click="activeTab = 'advanced'"
      >
        高级设置
      </button>
    </div>

    <!-- 基本设置 -->
    <div v-show="activeTab === 'basic'" class="tab-content">
      <!-- Firecrawl 配置 -->
      <div class="section">
        <h4 class="section-title">Firecrawl 配置</h4>
        <div class="form-row">
          <label class="form-label">API Key</label>
          <div class="input-group">
            <input
              :type="showFirecrawlKey ? 'text' : 'password'"
              class="form-input"
              :value="settings.firecrawl.apiKey"
              @input="store.setFirecrawlApiKey(($event.target as HTMLInputElement).value)"
              placeholder="fc-xxxxxxxxxxxxxxxx"
            />
            <button class="toggle-btn" @click="showFirecrawlKey = !showFirecrawlKey">
              {{ showFirecrawlKey ? '👁' : '👁‍🗨' }}
            </button>
            <button
              class="test-btn"
              :disabled="testingFirecrawl || !settings.firecrawl.apiKey"
              @click="handleTestFirecrawl"
            >
              {{ testingFirecrawl ? '测试中...' : '测试连接' }}
            </button>
          </div>
        </div>
      </div>

      <!-- AI Agent 配置 -->
      <div class="section">
        <h4 class="section-title">AI Agent 配置</h4>

        <div class="form-row">
          <label class="form-label">服务商</label>
          <select
            class="form-select"
            :value="settings.agent.provider"
            @change="store.setAgentProvider(($event.target as HTMLSelectElement).value)"
          >
            <option
              v-for="provider in WEB_IMPORT_AGENT_PROVIDERS"
              :key="provider.value"
              :value="provider.value"
            >
              {{ provider.label }}
            </option>
          </select>
        </div>

        <div class="form-row">
          <label class="form-label">API Key</label>
          <div class="input-group">
            <input
              :type="showAgentKey ? 'text' : 'password'"
              class="form-input"
              :value="settings.agent.apiKey"
              @input="store.setAgentApiKey(($event.target as HTMLInputElement).value)"
              placeholder="sk-xxxxxxxxxxxxxxxx"
            />
            <button class="toggle-btn" @click="showAgentKey = !showAgentKey">
              {{ showAgentKey ? '👁' : '👁‍🗨' }}
            </button>
          </div>
        </div>

        <div v-if="showCustomUrl" class="form-row">
          <label class="form-label">自定义 API 地址</label>
          <input
            type="url"
            class="form-input"
            :value="settings.agent.customBaseUrl"
            @input="store.setAgentBaseUrl(($event.target as HTMLInputElement).value)"
            placeholder="https://api.example.com/v1"
          />
        </div>

        <div class="form-row">
          <label class="form-label">模型名称</label>
          <input
            type="text"
            class="form-input"
            :value="settings.agent.modelName"
            @input="store.setAgentModelName(($event.target as HTMLInputElement).value)"
            placeholder="gpt-4o-mini"
          />
        </div>

        <div class="form-row inline">
          <label class="checkbox-label">
            <input
              type="checkbox"
              :checked="settings.agent.forceJsonOutput"
              @change="store.setAgentForceJson(($event.target as HTMLInputElement).checked)"
            />
            强制 JSON 格式
          </label>
          <label class="checkbox-label">
            <input
              type="checkbox"
              :checked="settings.agent.useStream"
              @change="store.setAgentUseStream(($event.target as HTMLInputElement).checked)"
            />
            流式调用
          </label>
        </div>

        <div class="form-row">
          <button
            class="test-btn full"
            :disabled="testingAgent || !settings.agent.apiKey"
            @click="handleTestAgent"
          >
            {{ testingAgent ? '测试中...' : '测试 Agent 连接' }}
          </button>
        </div>
      </div>

      <!-- 提取设置 -->
      <div class="section">
        <h4 class="section-title">
          提取设置
          <button class="reset-btn" @click="handleResetPrompt">重置为默认</button>
        </h4>

        <div class="form-row">
          <label class="form-label">提取提示词</label>
          <textarea
            class="form-textarea"
            :value="settings.extraction.prompt"
            @input="store.setExtractionPrompt(($event.target as HTMLTextAreaElement).value)"
            rows="6"
            placeholder="输入提取提示词..."
          ></textarea>
        </div>

        <div class="form-row">
          <label class="form-label">最大迭代次数</label>
          <input
            type="number"
            class="form-input small"
            :value="settings.extraction.maxIterations"
            @input="store.setExtractionMaxIterations(Number(($event.target as HTMLInputElement).value))"
            min="1"
            max="20"
          />
        </div>
      </div>

      <!-- 下载设置 -->
      <div class="section">
        <h4 class="section-title">下载设置</h4>

        <div class="form-grid">
          <div class="form-row">
            <label class="form-label">并发数</label>
            <input
              type="number"
              class="form-input small"
              :value="settings.download.concurrency"
              @input="store.setDownloadConcurrency(Number(($event.target as HTMLInputElement).value))"
              min="1"
              max="10"
            />
          </div>

          <div class="form-row">
            <label class="form-label">超时 (秒)</label>
            <input
              type="number"
              class="form-input small"
              :value="settings.download.timeout"
              @input="store.setDownloadTimeout(Number(($event.target as HTMLInputElement).value))"
              min="5"
              max="120"
            />
          </div>

          <div class="form-row">
            <label class="form-label">重试次数</label>
            <input
              type="number"
              class="form-input small"
              :value="settings.download.retries"
              @input="store.setDownloadRetries(Number(($event.target as HTMLInputElement).value))"
              min="0"
              max="5"
            />
          </div>

          <div class="form-row">
            <label class="form-label">下载间隔 (ms)</label>
            <input
              type="number"
              class="form-input small"
              :value="settings.download.delay"
              @input="store.setDownloadDelay(Number(($event.target as HTMLInputElement).value))"
              min="0"
              max="2000"
              step="100"
            />
          </div>
        </div>

        <div class="form-row">
          <label class="checkbox-label">
            <input
              type="checkbox"
              :checked="settings.download.useReferer"
              @change="store.setDownloadUseReferer(($event.target as HTMLInputElement).checked)"
            />
            自动添加 Referer
          </label>
        </div>
      </div>

      <!-- 界面设置 -->
      <div class="section">
        <h4 class="section-title">界面设置</h4>
        <div class="form-row inline">
          <label class="checkbox-label">
            <input
              type="checkbox"
              :checked="settings.ui.showAgentLogs"
              @change="store.setShowAgentLogs(($event.target as HTMLInputElement).checked)"
            />
            显示 AI 工作日志
          </label>
          <label class="checkbox-label">
            <input
              type="checkbox"
              :checked="settings.ui.autoImport"
              @change="store.setAutoImport(($event.target as HTMLInputElement).checked)"
            />
            提取后自动导入
          </label>
        </div>
      </div>
    </div>

    <!-- 图片预处理 -->
    <div v-show="activeTab === 'preprocess'" class="tab-content">
      <div class="section">
        <div class="form-row">
          <label class="checkbox-label">
            <input
              type="checkbox"
              :checked="settings.imagePreprocess.enabled"
              @change="store.setImagePreprocessEnabled(($event.target as HTMLInputElement).checked)"
            />
            启用图片预处理
          </label>
        </div>

        <template v-if="settings.imagePreprocess.enabled">
          <div class="form-row">
            <label class="checkbox-label">
              <input
                type="checkbox"
                :checked="settings.imagePreprocess.autoRotate"
                @change="store.setImageAutoRotate(($event.target as HTMLInputElement).checked)"
              />
              根据 EXIF 自动旋转
            </label>
          </div>

          <h5 class="subsection-title">压缩设置</h5>
          <div class="form-row">
            <label class="checkbox-label">
              <input
                type="checkbox"
                :checked="settings.imagePreprocess.compression.enabled"
                @change="store.setImageCompressionEnabled(($event.target as HTMLInputElement).checked)"
              />
              启用压缩
            </label>
          </div>

          <template v-if="settings.imagePreprocess.compression.enabled">
            <div class="form-grid">
              <div class="form-row">
                <label class="form-label">质量 (0-100)</label>
                <input
                  type="number"
                  class="form-input small"
                  :value="settings.imagePreprocess.compression.quality"
                  @input="store.setImageCompressionQuality(Number(($event.target as HTMLInputElement).value))"
                  min="1"
                  max="100"
                />
              </div>
              <div class="form-row">
                <label class="form-label">最大宽度 (0=不限)</label>
                <input
                  type="number"
                  class="form-input small"
                  :value="settings.imagePreprocess.compression.maxWidth"
                  @input="store.setImageMaxWidth(Number(($event.target as HTMLInputElement).value))"
                  min="0"
                />
              </div>
              <div class="form-row">
                <label class="form-label">最大高度 (0=不限)</label>
                <input
                  type="number"
                  class="form-input small"
                  :value="settings.imagePreprocess.compression.maxHeight"
                  @input="store.setImageMaxHeight(Number(($event.target as HTMLInputElement).value))"
                  min="0"
                />
              </div>
            </div>
          </template>

          <h5 class="subsection-title">格式转换</h5>
          <div class="form-row">
            <label class="checkbox-label">
              <input
                type="checkbox"
                :checked="settings.imagePreprocess.formatConvert.enabled"
                @change="store.setImageFormatConvertEnabled(($event.target as HTMLInputElement).checked)"
              />
              启用格式转换
            </label>
          </div>

          <div v-if="settings.imagePreprocess.formatConvert.enabled" class="form-row">
            <label class="form-label">目标格式</label>
            <select
              class="form-select"
              :value="settings.imagePreprocess.formatConvert.targetFormat"
              @change="store.setImageTargetFormat(($event.target as HTMLSelectElement).value as 'jpeg' | 'png' | 'webp' | 'original')"
            >
              <option value="original">保持原格式</option>
              <option value="jpeg">JPEG</option>
              <option value="png">PNG</option>
              <option value="webp">WebP</option>
            </select>
          </div>
        </template>
      </div>
    </div>

    <!-- 高级设置 -->
    <div v-show="activeTab === 'advanced'" class="tab-content">
      <div class="section">
        <h4 class="section-title">自定义请求头</h4>

        <div class="form-row">
          <label class="form-label">Cookie</label>
          <input
            type="text"
            class="form-input"
            :value="settings.advanced.customCookie"
            @input="store.setCustomCookie(($event.target as HTMLInputElement).value)"
            placeholder="name=value; name2=value2"
          />
        </div>

        <div class="form-row">
          <label class="form-label">Headers (JSON)</label>
          <textarea
            class="form-textarea"
            :value="settings.advanced.customHeaders"
            @input="store.setCustomHeaders(($event.target as HTMLTextAreaElement).value)"
            rows="3"
            placeholder='{"X-Custom-Header": "value"}'
          ></textarea>
        </div>

        <div class="form-row">
          <label class="checkbox-label">
            <input
              type="checkbox"
              :checked="settings.advanced.bypassProxy"
              @change="store.setBypassProxy(($event.target as HTMLInputElement).checked)"
            />
            绕过系统代理 (连接本地服务时使用)
          </label>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.web-import-settings {
  padding: 16px;
}

.tabs {
  display: flex;
  gap: 4px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--border-color, #eee);
  padding-bottom: 8px;
}

.tab {
  padding: 8px 16px;
  background: transparent;
  border: none;
  border-radius: 6px 6px 0 0;
  cursor: pointer;
  font-size: 14px;
  color: var(--text-secondary, #666);
  transition: all 0.2s;
}

.tab:hover {
  background: var(--bg-secondary, #f5f5f5);
}

.tab.active {
  background: var(--bg-secondary, #f5f5f5);
  color: var(--text-primary, #333);
  font-weight: 500;
}

.section {
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-color, #eee);
}

.section:last-child {
  border-bottom: none;
  margin-bottom: 0;
}

.section-title {
  margin: 0 0 12px 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary, #333);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.subsection-title {
  margin: 16px 0 8px 0;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary, #666);
}

.form-row {
  margin-bottom: 12px;
}

.form-row.inline {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 12px;
}

.form-label {
  display: block;
  margin-bottom: 4px;
  font-size: 13px;
  color: var(--text-secondary, #666);
}

.form-input,
.form-select,
.form-textarea {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
  background: var(--bg-primary, #fff);
  color: var(--text-primary, #333);
}

.form-input:focus,
.form-select:focus,
.form-textarea:focus {
  border-color: var(--primary-color, #4a90d9);
}

.form-input.small {
  width: 100px;
}

.form-textarea {
  resize: vertical;
  min-height: 80px;
}

.input-group {
  display: flex;
  gap: 8px;
}

.input-group .form-input {
  flex: 1;
}

.toggle-btn {
  padding: 8px 12px;
  background: var(--bg-secondary, #f5f5f5);
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  cursor: pointer;
}

.test-btn {
  padding: 8px 14px;
  background: var(--btn-secondary-bg, #f0f0f0);
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  white-space: nowrap;
  transition: all 0.2s;
}

.test-btn:hover:not(:disabled) {
  background: var(--btn-secondary-hover-bg, #e5e5e5);
}

.test-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.test-btn.full {
  width: 100%;
}

.reset-btn {
  padding: 4px 10px;
  background: transparent;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  color: var(--text-secondary, #666);
}

.reset-btn:hover {
  background: var(--bg-secondary, #f5f5f5);
}

.checkbox-label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  cursor: pointer;
  color: var(--text-primary, #333);
}

.checkbox-label input[type='checkbox'] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}
</style>
