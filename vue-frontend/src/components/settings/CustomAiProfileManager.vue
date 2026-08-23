<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import BaseModal from '@/components/common/BaseModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiPasswordField from '@/components/ui/UiPasswordField.vue'
import { useCustomAiProfileStore } from '@/stores/customAiProfileStore'
import {
  CUSTOM_AI_PROFILE_KIND_LABELS,
  type CustomAiProfile,
  type CustomAiProfileKind,
} from '@/types/customAiProfile'

const props = defineProps<{
  modelValue: boolean
  kind: CustomAiProfileKind
}>()

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  saved: [profile: CustomAiProfile]
}>()

const store = useCustomAiProfileStore()
const editingId = ref<string | null>(null)
const pendingDeleteId = ref<string | null>(null)
const draft = ref<Omit<CustomAiProfile, 'id'>>(emptyDraft())

const kindLabel = computed(() => CUSTOM_AI_PROFILE_KIND_LABELS[props.kind])
const profiles = computed(() => store.byKind(props.kind))
const canSave = computed(() => (
  draft.value.name.trim().length > 0
  && draft.value.baseUrl.trim().length > 0
  && draft.value.apiKey.trim().length > 0
  && draft.value.model.trim().length > 0
  && !store.isSaving
))

function emptyDraft(): Omit<CustomAiProfile, 'id'> {
  return {
    name: '',
    kind: props.kind,
    baseUrl: '',
    apiKey: '',
    model: '',
  }
}

function close(): void {
  if (store.isSaving) return
  emit('update:modelValue', false)
}

function startCreate(): void {
  editingId.value = null
  pendingDeleteId.value = null
  draft.value = emptyDraft()
}

function startEdit(profile: CustomAiProfile): void {
  editingId.value = profile.id
  pendingDeleteId.value = null
  draft.value = {
    name: profile.name,
    kind: profile.kind,
    baseUrl: profile.baseUrl,
    apiKey: profile.apiKey,
    model: profile.model,
  }
}

function updateDraft(
  field: 'name' | 'baseUrl' | 'apiKey' | 'model',
  value: string | number | boolean,
): void {
  if (typeof value === 'string') draft.value[field] = value
}

async function save(): Promise<void> {
  if (!canSave.value) return
  if (editingId.value) {
    const profile = { ...draft.value, id: editingId.value }
    if (await store.update(profile)) {
      const saved = store.profiles.find(item => item.id === profile.id)
      if (saved) emit('saved', saved)
      startCreate()
    }
    return
  }
  const profile = await store.create(draft.value)
  if (profile) {
    emit('saved', profile)
    startCreate()
  }
}

async function remove(profileId: string): Promise<void> {
  if (pendingDeleteId.value !== profileId) {
    pendingDeleteId.value = profileId
    return
  }
  if (await store.remove(profileId)) {
    pendingDeleteId.value = null
    if (editingId.value === profileId) startCreate()
  }
}

watch(
  () => props.modelValue,
  async (open) => {
    if (!open) return
    await store.load()
    startCreate()
  },
)

watch(
  () => props.kind,
  () => startCreate(),
)
</script>

<template>
  <BaseModal
    :model-value="modelValue"
    :title="`管理${kindLabel}自定义服务`"
    size="large"
    frame-variant="outlined"
    divider-variant="soft"
    max-height="86vh"
    :close-on-overlay="!store.isSaving"
    :close-on-esc="!store.isSaving"
    :show-close-button="!store.isSaving"
    @update:model-value="value => { if (!value) close() }"
  >
    <div class="custom-ai-profile-manager">
      <section class="custom-ai-profile-manager__list" aria-label="已保存配置">
        <div class="custom-ai-profile-manager__section-heading">
          <div>
            <h4>已保存配置</h4>
            <p>选择后会把 Base URL、API Key 和模型名应用到当前功能。</p>
          </div>
          <UiButton type="button" size="sm" variant="secondary" @click="startCreate">
            新增配置
          </UiButton>
        </div>

        <div v-if="profiles.length" class="custom-ai-profile-manager__records">
          <article
            v-for="profile in profiles"
            :key="profile.id"
            class="custom-ai-profile-manager__record"
          >
            <div class="custom-ai-profile-manager__record-copy">
              <strong>{{ profile.name }}</strong>
              <span>{{ profile.model }}</span>
              <small>{{ profile.baseUrl }}</small>
            </div>
            <ProductActionRow justify="end" :aria-label="`${profile.name} 操作`">
              <UiButton type="button" size="xs" variant="ghost" @click="startEdit(profile)">
                编辑
              </UiButton>
              <UiButton
                type="button"
                size="xs"
                :variant="pendingDeleteId === profile.id ? 'danger' : 'ghost'"
                :disabled="store.isSaving"
                @click="remove(profile.id)"
              >
                {{ pendingDeleteId === profile.id ? '确认删除' : '删除' }}
              </UiButton>
              <UiButton
                v-if="pendingDeleteId === profile.id"
                type="button"
                size="xs"
                variant="ghost"
                @click="pendingDeleteId = null"
              >
                取消
              </UiButton>
            </ProductActionRow>
          </article>
        </div>
        <p v-else class="custom-ai-profile-manager__empty">尚未保存此用途的配置。</p>
      </section>

      <section class="custom-ai-profile-manager__editor" aria-label="配置编辑器">
        <h4>{{ editingId ? '编辑配置' : '新增配置' }}</h4>
        <UiField variant="settings" label="配置名称" control-id="customAiProfileName">
          <UiInput
            id="customAiProfileName"
            :model-value="draft.name"
            placeholder="例如：公司中转服务"
            :disabled="store.isSaving"
            @update:model-value="updateDraft('name', $event)"
          />
        </UiField>
        <UiField variant="settings" label="Base URL" control-id="customAiProfileBaseUrl">
          <UiInput
            id="customAiProfileBaseUrl"
            :model-value="draft.baseUrl"
            placeholder="https://api.example.com/v1"
            :disabled="store.isSaving"
            @update:model-value="updateDraft('baseUrl', $event)"
          />
        </UiField>
        <UiField variant="settings" label="API Key" control-id="customAiProfileApiKey">
          <UiPasswordField
            input-id="customAiProfileApiKey"
            :model-value="draft.apiKey"
            placeholder="输入 API Key"
            :disabled="store.isSaving"
            show-label="显示自定义服务 API Key"
            hide-label="隐藏自定义服务 API Key"
            @update:model-value="value => updateDraft('apiKey', value)"
          />
        </UiField>
        <UiField variant="settings" label="模型名" control-id="customAiProfileModel">
          <UiInput
            id="customAiProfileModel"
            :model-value="draft.model"
            placeholder="输入模型名称"
            :disabled="store.isSaving"
            @update:model-value="updateDraft('model', $event)"
          />
        </UiField>

        <ProductStatusBanner v-if="store.error" tone="danger" role="alert">
          {{ store.error }}
        </ProductStatusBanner>

        <ProductActionRow justify="end" aria-label="配置编辑操作">
          <UiButton
            v-if="editingId"
            type="button"
            variant="secondary"
            :disabled="store.isSaving"
            @click="startCreate"
          >
            取消编辑
          </UiButton>
          <UiButton
            type="button"
            variant="primary"
            :disabled="!canSave"
            @click="save"
          >
            {{ store.isSaving ? '保存中…' : editingId ? '更新配置' : '保存配置' }}
          </UiButton>
        </ProductActionRow>
      </section>
    </div>
  </BaseModal>
</template>

<style scoped>
.custom-ai-profile-manager {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(280px, 0.9fr);
  gap: 20px;
}

.custom-ai-profile-manager__list,
.custom-ai-profile-manager__editor {
  min-width: 0;
}

.custom-ai-profile-manager__editor {
  padding: 18px;
  border: 1px solid var(--color-border-muted);
  border-radius: 12px;
  background: var(--color-surface-subtle);
}

.custom-ai-profile-manager__section-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 14px;
}

.custom-ai-profile-manager h4,
.custom-ai-profile-manager p {
  margin: 0;
}

.custom-ai-profile-manager h4 {
  color: var(--color-text-heading);
  font-size: 15px;
}

.custom-ai-profile-manager__section-heading p {
  margin-top: 4px;
  color: var(--color-text-supporting);
  font-size: 13px;
}

.custom-ai-profile-manager__records {
  display: grid;
  gap: 10px;
}

.custom-ai-profile-manager__record {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 14px;
  border: 1px solid var(--color-border-muted);
  border-radius: 10px;
  background: var(--color-surface-base);
}

.custom-ai-profile-manager__record-copy {
  display: grid;
  min-width: 0;
  gap: 3px;
}

.custom-ai-profile-manager__record-copy span,
.custom-ai-profile-manager__record-copy small {
  overflow: hidden;
  color: var(--color-text-supporting);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.custom-ai-profile-manager__empty {
  padding: 24px;
  border: 1px dashed var(--color-border-muted);
  border-radius: 10px;
  color: var(--color-text-supporting);
  text-align: center;
}

@media (--breakpoint-md-down) {
  .custom-ai-profile-manager {
    grid-template-columns: 1fr;
  }
}
</style>
