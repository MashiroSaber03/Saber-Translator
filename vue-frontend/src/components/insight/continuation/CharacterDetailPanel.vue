<template>
  <div class="character-detail-panel" :class="{ 'has-selection': !!character }">
    <div v-if="!character" class="empty-detail">
      <span>👈</span>
      <p>点击左侧角色查看详情</p>
    </div>
    
    <template v-else>
      <div class="detail-header">
        <div class="detail-main-info">
          <div class="detail-avatar">
            <img v-if="character.reference_image" :src="avatarUrl" alt="">
            <div v-else class="detail-avatar-placeholder">{{ character.name.charAt(0) }}</div>
          </div>
          <div class="detail-info">
            <h4>{{ character.name }}</h4>
            <p v-if="character.aliases && character.aliases.length > 0" class="detail-aliases">
              别名：{{ character.aliases.join('、') }}
            </p>
          </div>
        </div>
        <div class="detail-actions">
          <UiButton
            variant="toolbar"
            class="toggle-switch"
            :aria-label="`启用角色 ${character.name}`"
            :aria-pressed="character.enabled !== false"
            title="启用/禁用角色"
            @click="$emit('toggle-character', character.enabled === false)"
          >
            <span class="toggle-slider"></span>
          </UiButton>
          <UiButton variant="toolbar" class="icon-btn-lg" :aria-label="`编辑角色 ${character.name}`" @click="$emit('edit-character')" title="编辑角色">✏️</UiButton>
          <UiButton variant="danger" class="icon-btn-lg" :aria-label="`删除角色 ${character.name}`" @click="$emit('delete-character')" title="删除角色">🗑️</UiButton>
        </div>
      </div>
      
      <div class="forms-section">
        <div class="section-header">
          <h4>形态列表</h4>
          <UiButton variant="primary" @click="$emit('add-form')" size="sm">
            ➕ 新增形态
          </UiButton>
        </div>
        
        <div v-if="!character.forms || character.forms.length === 0" class="empty-forms">
          <p>暂无形态，点击"新增形态"添加</p>
        </div>
        
        <div v-else class="forms-grid">
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
  </div>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
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
  --character-detail-panel-card-border: #e0e4ff;
  --character-detail-panel-section-divider: #e5e7eb;
  --character-detail-panel-toggle-knob-shadow: rgba(0, 0, 0, .2);
  --character-detail-panel-action-shadow: rgba(0, 0, 0, .08);
  --character-detail-panel-action-hover-shadow: rgba(0, 0, 0, .12);
  --character-detail-panel-card-background-start: #fafbff;
  --character-detail-panel-card-background-end: #f5f7ff;
  --character-detail-panel-avatar-placeholder-background: #f0f0f0;
  --character-detail-panel-toggle-track-off: #cbd5e1;
  --character-detail-panel-toggle-track-on-start: #10b981;
  --character-detail-panel-toggle-track-on-end: #059669;
  --character-detail-panel-action-hover-background: #f0f2ff;
  --character-detail-panel-danger-hover-background: #fef2f2;
  --character-detail-panel-empty-text: #9ca3af;
  --character-detail-panel-title-text: #1a1a2e;
  --character-detail-panel-alias-text: #6b7280;
  --character-detail-panel-section-heading-text: #374151;
  --ui-button-padding: 6px 12px;
  --ui-button-radius: 6px;
  --ui-button-font-size: 13px;
  --ui-button-primary-background: var(--color-surface-brand);
  --ui-button-primary-hover-background: var(--color-surface-brand-strong);
  --ui-button-sm-padding: 6px 12px;
  --ui-button-sm-font-size: 13px;

  background: linear-gradient(135deg, var(--character-detail-panel-card-background-start) 0%, var(--character-detail-panel-card-background-end) 100%);
  border-radius: 16px;
  border: 1px solid var(--character-detail-panel-card-border);
  padding: 20px;
  display: flex;
  flex-direction: column;
  min-height: 280px;
}

.empty-detail {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--character-detail-panel-empty-text);
}

.empty-detail span {
  font-size: 48px;
  margin-bottom: 12px;
  opacity: 0.6;
}

.empty-detail p {
  margin: 0;
  font-size: 14px;
}

.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--character-detail-panel-section-divider);
  margin-bottom: 16px;
}

.detail-main-info {
  display: flex;
  gap: 14px;
  align-items: center;
}

.detail-avatar {
  width: 64px;
  height: 64px;
  border-radius: 12px;
  overflow: hidden;
  background: var(--character-detail-panel-avatar-placeholder-background);
  flex-shrink: 0;
}

.detail-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.detail-avatar-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, var(--color-action-brand) 0%, var(--color-action-brand-strong) 100%);
  color: var(--color-text-inverse);
  font-size: 24px;
  font-weight: 600;
}

.detail-info h4 {
  margin: 0 0 4px;
  font-size: 18px;
  font-weight: 600;
  color: var(--character-detail-panel-title-text);
}

.detail-aliases {
  margin: 0;
  font-size: 13px;
  color: var(--character-detail-panel-alias-text);
}

.detail-actions {
  display: flex;
  gap: 8px;
}

.toggle-switch {
  position: relative;
  display: inline-flex;
  width: 40px;
  height: 22px;
  padding: 0;
  border: 0;
  border-radius: 22px;
  background: transparent;
  cursor: pointer;
}

.toggle-slider {
  position: absolute;
  cursor: pointer;
  inset: 0;
  background-color: var(--character-detail-panel-toggle-track-off);
  transition: 0.3s;
  border-radius: 22px;
}

.toggle-slider::before {
  position: absolute;
  content: "";
  height: 16px;
  width: 16px;
  left: 3px;
  bottom: 3px;
  background-color: var(--color-surface-base);
  transition: 0.3s;
  border-radius: 50%;
  box-shadow: 0 1px 3px var(--character-detail-panel-toggle-knob-shadow);
}

.toggle-switch[aria-pressed='true'] .toggle-slider {
  background: linear-gradient(135deg, var(--character-detail-panel-toggle-track-on-start), var(--character-detail-panel-toggle-track-on-end));
}

.toggle-switch[aria-pressed='true'] .toggle-slider::before {
  transform: translateX(18px);
}

.icon-btn-lg {
  width: 40px;
  height: 40px;
  border: none;
  background: var(--color-surface-base);
  border-radius: 10px;
  cursor: pointer;
  font-size: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  box-shadow: 0 2px 6px var(--character-detail-panel-action-shadow);
}

.icon-btn-lg:hover {
  background: var(--character-detail-panel-action-hover-background);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px var(--character-detail-panel-action-hover-shadow);
}

.icon-btn-lg.danger:hover {
  background: var(--character-detail-panel-danger-hover-background);
}

.forms-section {
  flex: 1;
}

.forms-section h4 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--character-detail-panel-section-heading-text);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.empty-forms {
  text-align: center;
  padding: 40px 20px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.empty-forms p {
  margin: 0;
}

.forms-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 14px;
}
</style>
