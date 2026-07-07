<template>
  <div class="proofreading-settings">
    <ProductFormSection>
      <template #title>AI校对设置</template>
      <UiField
        variant="settings"
        control="checkbox"
        hint="翻译完成后自动进行AI校对"
      >
        <UiCheckbox v-model="isProofreadingEnabled" label="启用AI校对" />
      </UiField>
      <UiField variant="settings" label="全局重试次数" control-id="settingsProofreadingMaxRetries">
        <UiNumberField
          input-id="settingsProofreadingMaxRetries"
          v-model="proofreadingMaxRetries"
          :min="0"
          :max="10"
          :step="1"
        />
      </UiField>
    </ProductFormSection>

    <ProductFormSection v-show="isProofreadingEnabled">
      <template #title>
        校对轮次配置
        <UiButton
          variant="secondary"
          class="proofreading-settings__add-round-action"
          @click="addRound"
          size="sm"
        >
          <UiIcon name="plus" />
          <span>添加轮次</span>
        </UiButton>
      </template>

      <div v-for="(round, index) in proofreadingRounds" :key="index" class="proofreading-settings__round">
        <div class="proofreading-settings__round-header">
          <span class="proofreading-settings__round-title">轮次 {{ index + 1 }}: {{ round.name || '未命名' }}</span>
          <UiButton
            variant="danger"
            class="proofreading-settings__round-delete-action"
            @click="removeRound(index)"
            :disabled="proofreadingRounds.length <= 1"
            size="sm"
          >
            删除
          </UiButton>
        </div>

        <div class="proofreading-settings__round-content">
          <UiField
            variant="settings"
            label="轮次名称"
            :control-id="roundFieldId(index, 'Name')"
          >
            <UiInput
              type="text"
              :id="roundFieldId(index, 'Name')"
              v-model="round.name"
              placeholder="如: 第一轮校对"
            />
          </UiField>

          <UiFormGrid>
            <UiField
              variant="settings"
              label="服务商"
              :control-id="roundFieldId(index, 'Provider')"
            >
              <UiSelect
                :id="roundFieldId(index, 'Provider')"
                :model-value="round.provider"
                :options="providerOptions"
                @change="(value: string | number) => handleRoundProviderChange(index, value)"
              />
            </UiField>
            <UiField
              v-show="providerRequiresApiKey(round.provider)"
              variant="settings"
              label="API Key"
              :control-id="roundFieldId(index, 'ApiKey')"
            >
              <UiPasswordField
                :input-id="roundFieldId(index, 'ApiKey')"
                v-model="round.apiKey"
                placeholder="请输入API Key"
                :show-label="`显示${round.name} API Key`"
                :hide-label="`隐藏${round.name} API Key`"
              />
            </UiField>
          </UiFormGrid>

          <UiField
            v-show="providerRequiresBaseUrl(round.provider)"
            variant="settings"
            label="Base URL"
            :control-id="roundFieldId(index, 'BaseUrl')"
          >
            <UiInput
              type="text"
              :id="roundFieldId(index, 'BaseUrl')"
              v-model="round.customBaseUrl"
              placeholder="例如: https://api.example.com/v1"
            />
          </UiField>

          <UiField
            variant="settings"
            label="模型名称"
            :control-id="roundFieldId(index, 'ModelName')"
          >
            <UiModelPicker
              :input-id="roundFieldId(index, 'ModelName')"
              v-model="round.modelName"
              placeholder="请输入模型名称"
              fetch-variant="primary"
              :fetching="Boolean(roundFetchingStates[index])"
              :fetch-disabled="Boolean(roundFetchingStates[index])"
              :options="getRoundModelOptions(index)"
              :model-count="roundModelLists[index]?.length || 0"
              @fetch="fetchRoundModels(index)"
            />
          </UiField>

          <UiField variant="settings">
            <UiButton
              variant="secondary"
              block
              @click="testRoundConnection(index)"
              :disabled="roundTestingStates[index]"
            >
              <span v-if="roundTestingStates[index]">测试中...</span>
              <template v-else>
                <UiIcon name="link" />
                <span>测试连接</span>
              </template>
            </UiButton>
          </UiField>

          <UiFormGrid>
            <UiField
              variant="settings"
              label="批次大小"
              :control-id="roundFieldId(index, 'BatchSize')"
            >
              <UiNumberField
                :input-id="roundFieldId(index, 'BatchSize')"
                v-model="round.batchSize"
                :min="1"
                :max="10"
                :step="1"
              />
            </UiField>
            <UiField
              variant="settings"
              label="RPM限制"
              :control-id="roundFieldId(index, 'RpmLimit')"
            >
              <UiNumberField
                :input-id="roundFieldId(index, 'RpmLimit')"
                v-model="round.openaiOptions.execution.rpmLimit"
                :min="0"
                :step="1"
              />
            </UiField>
          </UiFormGrid>

          <UiFormGrid>
            <UiField
              variant="settings"
              label="业务重试"
              :control-id="roundFieldId(index, 'BusinessRetries')"
            >
              <UiNumberField
                :input-id="roundFieldId(index, 'BusinessRetries')"
                v-model="round.openaiOptions.execution.businessRetries"
                :min="0"
                :max="10"
                :step="1"
              />
            </UiField>
            <UiField
              variant="settings"
              label="传输重试"
              :control-id="roundFieldId(index, 'TransportRetries')"
            >
              <UiNumberField
                :input-id="roundFieldId(index, 'TransportRetries')"
                v-model="round.openaiOptions.execution.transportRetries"
                :min="0"
                :max="10"
                :step="1"
              />
            </UiField>
          </UiFormGrid>
          <UiFormGrid>
            <UiField
              variant="settings"
              control="checkbox"
              hint="使用 response_format: json_object"
            >
              <UiCheckbox v-model="round.openaiOptions.request.forceJsonOutput" label="强制JSON输出" />
            </UiField>
            <UiField
              variant="settings"
              control="checkbox"
              hint="使用流式API调用，避免超时"
            >
              <UiCheckbox v-model="round.openaiOptions.execution.useStream" label="流式调用" />
            </UiField>
          </UiFormGrid>
          <UiField variant="settings">
            <OpenAIExtraBodyEditor v-model="round.openaiOptions.request.extraBody" />
          </UiField>

          <UiField
            variant="settings"
            label="校对提示词"
            :control-id="roundFieldId(index, 'Prompt')"
          >
            <UiTextarea
              :id="roundFieldId(index, 'Prompt')"
              v-model="round.prompt"
              variant="panel"
              rows="4"
              placeholder="校对提示词"
            />
            <SavedPromptsPicker
              prompt-type="proofreading"
              @select="(content, name) => handleProofreadingPromptSelect(index, content, name)"
            />
            <ProductActionRow aria-label="校对提示词操作" justify="start">
              <UiButton variant="secondary" @click="resetRoundPrompt(index)" size="sm">
                重置为默认
              </UiButton>
            </ProductActionRow>
          </UiField>
        </div>
      </div>
    </ProductFormSection>
  </div>
</template>

<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiPasswordField from '@/components/ui/UiPasswordField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import { ref, computed, watch } from 'vue'
import {
  getProviderOptionsForCapability,
  providerRequiresApiKey,
  providerSupportsCapability,
  providerRequiresBaseUrl
} from '@/config/aiProviders'
import { useSettingsStore } from '@/stores/settings'
import { configApi } from '@/api/config'
import { useToast } from '@/utils/toast'
import { DEFAULT_PROOFREADING_PROMPT } from '@/constants'
import type { ProofreadingRound } from '@/types/settings'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'
import { useKeyedLatestRequestGuard } from '@/composables/useLatestRequestGuard'

const providerOptions = getProviderOptionsForCapability('hqTranslation')

const settingsStore = useSettingsStore()
const toast = useToast()

const roundFetchingStates = ref<Record<number, boolean>>({})
const roundTestingStates = ref<Record<number, boolean>>({})
const roundModelLists = ref<Record<number, string[]>>({})
const roundModelFetchGuard = useKeyedLatestRequestGuard<number>()

const proofreadingRounds = computed(() => settingsStore.settings.proofreading.rounds)
const proofreadingMaxRetries = computed({
  get: () => settingsStore.settings.proofreading.maxRetries,
  set: (val: number) => settingsStore.setProofreadingMaxRetries(val)
})
const isProofreadingEnabled = computed({
  get: () => settingsStore.settings.proofreading.enabled,
  set: (val: boolean) => settingsStore.setProofreadingEnabled(val)
})

function roundFieldId(index: number, field: string) {
  return `proofreadingRound${index}${field}`
}

watch(
  () => settingsStore.settings.proofreading.rounds,
  () => {
    settingsStore.saveToStorage()
  },
  { deep: true }
)

function getRoundModelOptions(index: number) {
  const models = roundModelLists.value[index] || []
  const options = [{ label: '-- 选择模型 --', value: '' }]
  models.forEach(m => options.push({ label: m, value: m }))
  return options
}

function invalidateRoundModelFetch(index: number) {
  roundModelFetchGuard.invalidate(index)
  roundFetchingStates.value[index] = false
}

function handleRoundProviderChange(index: number, value: string | number) {
  const round = proofreadingRounds.value[index]
  if (!round) return
  invalidateRoundModelFetch(index)
  round.provider = String(value) as ProofreadingRound['provider']
  roundModelLists.value[index] = []
}

function isCurrentRoundModelFetch(index: number, requestId: number, provider: string, apiKey: string | undefined, baseUrl: string | undefined) {
  return roundModelFetchGuard.isCurrent(index, requestId, () => {
    const round = proofreadingRounds.value[index]
    return Boolean(
      round &&
      round.provider === provider &&
      round.apiKey?.trim() === apiKey &&
      round.customBaseUrl?.trim() === baseUrl
    )
  })
}

async function fetchRoundModels(index: number) {
  const round = proofreadingRounds.value[index]
  if (!round) return

  const provider = round.provider
  const apiKey = round.apiKey?.trim()
  const baseUrl = round.customBaseUrl?.trim()

  if (providerRequiresApiKey(provider) && !apiKey) {
    toast.warning('请先填写 API Key')
    return
  }

  if (!providerSupportsCapability(provider, 'modelFetch')) {
    toast.warning('当前服务商不支持获取模型列表')
    return
  }

  const requestId = roundModelFetchGuard.next(index)
  roundFetchingStates.value[index] = true
  try {
    const result = await configApi.fetchModels(provider, apiKey, baseUrl)
    if (!isCurrentRoundModelFetch(index, requestId, provider, apiKey, baseUrl)) return
    if (result.success && result.models && result.models.length > 0) {
      roundModelLists.value[index] = result.models.map(m => m.id)
      toast.success(`轮次 ${index + 1}: 获取到 ${result.models.length} 个模型`)
    } else {
      toast.warning(result.message || '未获取到可用模型')
    }
  } catch (error: unknown) {
    if (!isCurrentRoundModelFetch(index, requestId, provider, apiKey, baseUrl)) return
    const errorMessage = error instanceof Error ? error.message : '获取模型列表失败'
    toast.error(errorMessage)
  } finally {
    if (roundModelFetchGuard.isCurrent(index, requestId)) {
      roundFetchingStates.value[index] = false
    }
  }
}

async function testRoundConnection(index: number) {
  const round = proofreadingRounds.value[index]
  if (!round) return

  const provider = round.provider
  const apiKey = round.apiKey?.trim()
  const modelName = round.modelName?.trim()
  const baseUrl = round.customBaseUrl?.trim()

  if (providerRequiresApiKey(provider) && !apiKey) {
    toast.warning('请先填写 API Key')
    return
  }

  if (!modelName) {
    toast.warning('请填写模型名称')
    return
  }

  roundTestingStates.value[index] = true
  toast.info(`正在测试轮次 ${index + 1} 的连接...`)

  try {
    const result = await configApi.testAiTranslateConnection({
      provider,
      apiKey,
      modelName,
      baseUrl
    })

    if (result.success) {
      toast.success(result.message || '连接成功!')
    } else {
      toast.error(result.message || result.error || '连接失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '连接测试失败'
    toast.error(errorMessage)
  } finally {
    roundTestingStates.value[index] = false
  }
}

function addRound() {
  const newRound: ProofreadingRound = {
    name: `第${proofreadingRounds.value.length + 1}轮校对`,
    provider: 'siliconflow',
    apiKey: '',
    modelName: '',
    customBaseUrl: '',
    openaiOptions: {
      request: {
        forceJsonOutput: false
      },
      execution: {
        useStream: true,
        rpmLimit: 7,
        transportRetries: 1,
        businessRetries: settingsStore.settings.proofreading.maxRetries
      }
    },
    batchSize: 3,
    prompt: DEFAULT_PROOFREADING_PROMPT,
  }
  settingsStore.addProofreadingRound(newRound)
  toast.success('已添加新的校对轮次')
}

function removeRound(index: number) {
  if (proofreadingRounds.value.length <= 1) {
    toast.warning('至少需要保留一个校对轮次')
    return
  }
  settingsStore.removeProofreadingRound(index)
  toast.success('已删除校对轮次')
}

function resetRoundPrompt(index: number) {
  settingsStore.updateProofreadingRound(index, { prompt: DEFAULT_PROOFREADING_PROMPT })
  toast.success('已重置为默认提示词')
}

function handleProofreadingPromptSelect(index: number, content: string, name: string) {
  settingsStore.updateProofreadingRound(index, { prompt: content })
  toast.success(`已应用提示词: ${name}`)
}
</script>

<style scoped>
.proofreading-settings__round {
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  margin-bottom: 15px;
  overflow: hidden;
}

.proofreading-settings__round-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 15px;
  background: var(--color-surface-subtle);
  border-bottom: 1px solid var(--color-border-muted);
}

.proofreading-settings__round-title {
  font-weight: 500;
}

.proofreading-settings__round-content {
  padding: 15px;
}
</style>
