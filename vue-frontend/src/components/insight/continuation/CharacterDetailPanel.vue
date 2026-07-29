<template>
  <ProductDetailPanel
    class="character-detail-panel"
    :class="{ 'character-detail-panel--has-selection': !!character }"
    :aria-label="character ? `角色 ${character.name} 详情` : '角色详情'"
  >
    <ProductStatusBanner
      v-if="!character"
      class="character-detail-panel__empty-status"
      tone="neutral"
      role="note"
      icon-name="users"
      title="选择角色"
    >
      点击左侧角色查看详情
    </ProductStatusBanner>

    <template v-else>
      <div class="character-detail-panel__header">
        <div class="character-detail-panel__main-info">
          <ProductAvatar
            :image-src="character.reference_image ? avatarUrl : ''"
            :label="`角色 ${character.name} 头像`"
            :fallback-text="character.name"
            size="lg"
            shape="rounded"
          />
          <div class="character-detail-panel__info">
            <h4 class="character-detail-panel__title">{{ character.name }}</h4>
            <p v-if="character.aliases && character.aliases.length > 0" class="character-detail-panel__aliases">
              别名：{{ character.aliases.join('、') }}
            </p>
          </div>
        </div>
        <ProductActionRow
          class="character-detail-panel__actions"
          justify="end"
          :aria-label="`${character.name} 角色操作`"
        >
          <UiSwitch
            :model-value="character.enabled !== false"
            :ariaLabel="`启用角色 ${character.name}`"
            title="启用/禁用角色"
            @change="$emit('toggle-character', $event)"
          />
          <UiIconButton variant="plain" :label="`编辑角色 ${character.name}`" @click="$emit('edit-character')">
            <UiIcon name="pencil" size="16" />
          </UiIconButton>
          <UiIconButton variant="danger" :label="`删除角色 ${character.name}`" @click="$emit('delete-character')">
            <UiIcon name="trash" size="16" />
          </UiIconButton>
        </ProductActionRow>
      </div>

      <div class="character-detail-panel__forms-section">
        <ProductSectionHeader title="形态列表">
          <template #actions>
            <UiButton variant="primary" @click="$emit('add-form')" size="sm">
              <UiIcon name="plus" size="14" />
              <span>新增形态</span>
            </UiButton>
          </template>
        </ProductSectionHeader>

        <ProductStatusBanner
          v-if="!character.forms || character.forms.length === 0"
          class="character-detail-panel__empty-forms-status"
          tone="neutral"
          role="note"
          icon-name="list"
          title="暂无形态"
        >
          点击“新增形态”添加角色形态。
        </ProductStatusBanner>

        <div v-else class="character-detail-panel__forms-grid">
          <FormTile
            v-for="form in character.forms"
            :key="form.form_id"
            :form="form"
            :character-name="character.name"
            :form-image-url="getFormImageUrl(form.form_id)"
            @edit="$emit('edit-form', form)"
            @delete="$emit('delete-form', form)"
            @upload-image="(file) => $emit('upload-form-image', form.form_id, file)"
            @delete-image="$emit('delete-form-image', form.form_id)"
            @generate-orthographic="$emit('generate-orthographic', form.form_id, form.form_name)"
            @toggle-enabled="(enabled) => $emit('toggle-form-enabled', form.form_id, enabled)"
          />
        </div>
      </div>
    </template>
  </ProductDetailPanel>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiSwitch from '@/components/ui/UiSwitch.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductAvatar from '@/components/product/ProductAvatar.vue'
import ProductDetailPanel from '@/components/product/ProductDetailPanel.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import type { CharacterProfile, CharacterForm } from '@/api/continuation'
import FormTile from './FormTile.vue'

defineProps<{
  character: CharacterProfile | null
  avatarUrl: string
  getFormImageUrl: (formId: string) => string
}>()

defineEmits<{
  'toggle-character': [enabled: boolean]
  'edit-character': []
  'delete-character': []
  'add-form': []
  'edit-form': [form: CharacterForm]
  'delete-form': [form: CharacterForm]
  'upload-form-image': [formId: string, file: File]
  'delete-form-image': [formId: string]
  'generate-orthographic': [formId: string, formName: string]
  'toggle-form-enabled': [formId: string, enabled: boolean]
}>()
</script>

<style scoped>
.character-detail-panel {
  display: flex;
  flex-direction: column;
  margin-bottom: 0;
  min-height: 280px;
}

.character-detail-panel__empty-status {
  flex: 1;
  justify-content: center;
}

.character-detail-panel__header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--color-border-muted);
  margin-bottom: 16px;
}

.character-detail-panel__main-info {
  display: flex;
  gap: 14px;
  align-items: center;
}

.character-detail-panel__title {
  margin: 0 0 4px;
  font-size: 18px;
  font-weight: 600;
  color: var(--color-text-strong);
}

.character-detail-panel__aliases {
  margin: 0;
  font-size: 13px;
  color: var(--color-text-supporting);
}

.character-detail-panel__actions {
  flex: 0 0 auto;
  gap: 8px;
}

.character-detail-panel__forms-section {
  flex: 1;
}

.character-detail-panel__empty-forms-status {
  margin-top: 12px;
}

.character-detail-panel__forms-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 14px;
}
</style>
