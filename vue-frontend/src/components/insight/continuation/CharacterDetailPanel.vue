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
          <label class="toggle-switch" title="启用/禁用角色">
            <input 
              type="checkbox" 
              :checked="character.enabled !== false"
              @change="$emit('toggle-character', ($event.target as HTMLInputElement).checked)"
            >
            <span class="toggle-slider"></span>
          </label>
          <button class="icon-btn-lg" @click="$emit('edit-character')" title="编辑角色">✏️</button>
          <button class="icon-btn-lg danger" @click="$emit('delete-character')" title="删除角色">🗑️</button>
        </div>
      </div>
      
      <div class="forms-section">
        <div class="section-header">
          <h4>形态列表</h4>
          <button class="btn small primary" @click="$emit('add-form')">
            ➕ 新增形态
          </button>
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
import type { CharacterProfile, CharacterForm } from '@/api/continuation'
import FormTile from './FormTile.vue'

const props = defineProps<{
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
  background: linear-gradient(135deg, #fafbff 0%, #f5f7ff 100%);
  border-radius: 16px;
  border: 1px solid #e0e4ff;
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
  color: #9ca3af;
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
  border-bottom: 1px solid #e5e7eb;
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
  background: #f0f0f0;
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
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-size: 24px;
  font-weight: 600;
}

.detail-info h4 {
  margin: 0 0 4px;
  font-size: 18px;
  font-weight: 600;
  color: #1a1a2e;
}

.detail-aliases {
  margin: 0;
  font-size: 13px;
  color: #6b7280;
}

.detail-actions {
  display: flex;
  gap: 8px;
}

/* Toggle Switch */
.toggle-switch {
  position: relative;
  display: inline-block;
  width: 40px;
  height: 22px;
  cursor: pointer;
}

.toggle-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-switch .toggle-slider {
  position: absolute;
  cursor: pointer;
  inset: 0;
  background-color: #cbd5e1;
  transition: 0.3s;
  border-radius: 22px;
}

.toggle-switch .toggle-slider::before {
  position: absolute;
  content: "";
  height: 16px;
  width: 16px;
  left: 3px;
  bottom: 3px;
  background-color: white;
  transition: 0.3s;
  border-radius: 50%;
  box-shadow: 0 1px 3px rgb(0, 0, 0, 0.2);
}

.toggle-switch input:checked + .toggle-slider {
  background: linear-gradient(135deg, #10b981, #059669);
}

.toggle-switch input:checked + .toggle-slider::before {
  transform: translateX(18px);
}

.icon-btn-lg {
  width: 40px;
  height: 40px;
  border: none;
  background: #fff;
  border-radius: 10px;
  cursor: pointer;
  font-size: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  box-shadow: 0 2px 6px rgb(0, 0, 0, 0.08);
}

.icon-btn-lg:hover {
  background: #f0f2ff;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgb(0, 0, 0, 0.12);
}

.icon-btn-lg.danger:hover {
  background: #fef2f2;
}

.forms-section {
  flex: 1;
}

.forms-section h4 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: #374151;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.btn {
  padding: 6px 12px;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn.primary {
  background: var(--primary, #6366f1);
  color: white;
}

.btn.primary:hover {
  background: var(--primary-dark, #4f46e5);
}

.btn.small {
  padding: 6px 12px;
  font-size: 13px;
}

.empty-forms {
  text-align: center;
  padding: 40px 20px;
  color: var(--text-secondary, #666);
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
