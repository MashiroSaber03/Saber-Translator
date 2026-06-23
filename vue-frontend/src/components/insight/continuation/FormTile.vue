<template>
  <div class="form-tile" :class="{ disabled: form.enabled === false }">
    <!-- 图片区域 -->
    <div class="form-image-section">
      <img v-if="form.reference_image" :src="formImageUrl" alt="">
      <div v-else class="image-placeholder">
        <span class="placeholder-icon">📷</span>
        <p class="placeholder-text">未上传参考图</p>
      </div>
      <label class="upload-overlay">
        <span class="upload-text">{{ form.reference_image ? '更换图片' : '上传图片' }}</span>
        <UiFileInput
          accept="image/*"
          hidden
          :aria-label="`上传 ${characterName} ${form.form_name} 参考图`"
          @change="handleUpload"
        />
      </label>
    </div>
    
    <!-- 信息区域 -->
    <div class="form-content">
      <div class="form-header">
        <h4 class="form-title">{{ form.form_name }}</h4>
        <span v-if="form.enabled === false" class="status-badge disabled">已禁用</span>
      </div>
      <p v-if="form.description" class="form-description">{{ form.description }}</p>
    </div>
    
    <!-- 操作区域 -->
    <div class="form-actions">
      <div class="action-row">
        <label class="toggle-control" :title="form.enabled !== false ? '点击禁用' : '点击启用'">
          <UiInput 
            type="checkbox" 
            :aria-label="`启用 ${characterName} ${form.form_name}`"
            :checked="form.enabled !== false"
            @change="$emit('toggle-enabled', ($event.target as HTMLInputElement).checked)"
          />
          <span class="toggle-track"></span>
        </label>
        <UiButton variant="toolbar" class="action-btn generate-btn" :aria-label="`生成 ${characterName} ${form.form_name} 三视图`" @click="$emit('generate-orthographic')" title="生成三视图">
          <span>🎨</span>
        </UiButton>
        <UiButton variant="toolbar" v-if="form.reference_image" class="action-btn delete-btn" :aria-label="`删除 ${characterName} ${form.form_name} 参考图`" @click="$emit('delete-image')" title="删除图片">
          <span>🗑️</span>
        </UiButton>
      </div>
      <div class="action-row secondary">
        <UiButton variant="toolbar" class="icon-btn edit-btn" :aria-label="`编辑 ${characterName} ${form.form_name}`" @click="$emit('edit')" title="编辑形态">✏️</UiButton>
        <UiButton variant="toolbar" class="icon-btn delete-btn" :aria-label="`删除 ${characterName} ${form.form_name}`" @click="$emit('delete')" title="删除形态">🗑️</UiButton>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import type { CharacterForm } from '@/api/continuation'

defineProps<{
  form: CharacterForm
  characterName: string
  formImageUrl: string
}>()

const emit = defineEmits<{
  'edit': []
  'delete': []
  'upload-image': [file: File]
  'delete-image': []
  'generate-orthographic': []
  'toggle-enabled': [enabled: boolean]
}>()

function handleUpload(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files?.length) return
  
  const file = input.files[0]
  if (!file) return
  
  emit('upload-image', file)
  input.value = ''
}
</script>

<style scoped>
/* 卡片容器 */
.form-tile {
  /* owner tokens: form-tile */
  --form-tile-border-default: #e8eaf6;
  --form-tile-border-strong: #c7d2fe;
  --form-tile-border-muted: #cbd5e1;
  --form-tile-border-subtle: #a5b4fc;
  --form-tile-border-hover: #818cf8;
  --form-tile-border-active: #fecaca;
  --form-tile-shadow-default: rgba(99, 102, 241, .08);
  --form-tile-shadow-raised: rgba(0, 0, 0, .1);
  --form-tile-shadow-floating: rgba(0, 0, 0, .2);
  --form-tile-surface-base: #f8f9ff;
  --form-tile-surface-raised: #f5f7ff;
  --form-tile-surface-muted: #eef2ff;
  --form-tile-surface-subtle: rgba(99, 102, 241, .92);
  --form-tile-surface-hover: rgba(124, 58, 237, .92);
  --form-tile-surface-active: #fee2e2;
  --form-tile-surface-selected: #fecaca;
  --form-tile-surface-overlay: #fafbff;
  --form-tile-surface-inverse: #cbd5e1;
  --form-tile-surface-contrast: #94a3b8;
  --form-tile-surface-tint: #10b981;
  --form-tile-surface-soft: #059669;
  --form-tile-surface-strong: #fef2f2;
  --form-tile-text-primary: #9ca3af;
  --form-tile-text-secondary: #1e293b;
  --form-tile-text-muted: #dc2626;
  --form-tile-text-subtle: #64748b;

  background: linear-gradient(135deg, var(--color-surface-base) 0%, var(--form-tile-surface-base) 100%);
  border-radius: 16px;
  overflow: hidden;
  border: 1.5px solid var(--form-tile-border-default);
  box-shadow: 0 2px 8px var(--form-tile-shadow-default);
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  flex-direction: column;
}

.form-tile:hover {
  border-color: var(--form-tile-border-strong);
  box-shadow: 0 8px 24px var(--color-focus-brand-subtle);
  transform: translateY(-2px);
}

.form-tile.disabled {
  opacity: 0.6;
  filter: grayscale(60%);
}

.form-tile.disabled:hover {
  transform: none;
}

/* 图片区域 */
.form-image-section {
  aspect-ratio: 1;
  position: relative;
  background: linear-gradient(135deg, var(--form-tile-surface-raised) 0%, var(--form-tile-surface-muted) 100%);
  overflow: hidden;
}

.form-image-section img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--form-tile-text-primary);
}

.placeholder-icon {
  font-size: 48px;
  margin-bottom: 8px;
  opacity: 0.5;
}

.placeholder-text {
  margin: 0;
  font-size: 12px;
  font-weight: 500;
  color: var(--form-tile-text-primary);
}

/* 上传遮罩 */
.upload-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, var(--form-tile-surface-subtle), var(--form-tile-surface-hover));
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.25s ease;
  cursor: pointer;
}

.upload-text {
  color: white;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.3px;
  text-shadow: 0 1px 2px var(--form-tile-shadow-raised);
}

.form-image-section:hover .upload-overlay {
  opacity: 1;
}

/* 内容区域 */
.form-content {
  padding: 14px 12px 12px;
  flex: 1;
  display: flex;
  flex-direction: column;
}

.form-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 6px;
}

.form-title {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--form-tile-text-secondary);
  flex: 1;
  line-height: 1.3;
}

.status-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.3px;
}

.status-badge.disabled {
  background: linear-gradient(135deg, var(--form-tile-surface-active), var(--form-tile-surface-selected));
  color: var(--form-tile-text-muted);
}

.form-description {
  margin: 0;
  font-size: 11px;
  color: var(--form-tile-text-subtle);
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* 操作区域 */
.form-actions {
  padding: 10px 12px;
  background: linear-gradient(to bottom, var(--form-tile-surface-overlay), var(--form-tile-surface-base));
  border-top: 1px solid var(--form-tile-border-default);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.action-row {
  display: flex;
  gap: 6px;
  align-items: center;
}

.action-row.secondary {
  padding-top: 2px;
}

/* Toggle开关 */
.toggle-control {
  position: relative;
  display: inline-block;
  width: 32px;
  height: 18px;
  cursor: pointer;
  flex-shrink: 0;
}

.toggle-control input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-track {
  position: absolute;
  cursor: pointer;
  inset: 0;
  background: linear-gradient(135deg, var(--form-tile-surface-inverse), var(--form-tile-surface-contrast));
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border-radius: 18px;
}

.toggle-track::before {
  position: absolute;
  content: "";
  height: 14px;
  width: 14px;
  left: 2px;
  bottom: 2px;
  background: white;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border-radius: 50%;
  box-shadow: 0 2px 4px var(--form-tile-shadow-floating);
}

.toggle-control input:checked + .toggle-track {
  background: linear-gradient(135deg, var(--form-tile-surface-tint), var(--form-tile-surface-soft));
}

.toggle-control input:checked + .toggle-track::before {
  transform: translateX(14px);
}

/* 图标按钮 */
.action-btn {
  flex: 1;
  height: 32px;
  padding: 0 10px;
  border: 1.5px solid var(--color-border-muted);
  background: white;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.action-btn:hover {
  background: var(--color-surface-quiet);
  border-color: var(--form-tile-border-muted);
}

.action-btn.generate-btn {
  border-color: var(--form-tile-border-subtle);
  color: var(--color-text-brand);
}

.action-btn.generate-btn:hover {
  background: var(--form-tile-surface-muted);
  border-color: var(--form-tile-border-hover);
}

.action-btn.delete-btn:hover {
  background: var(--form-tile-surface-strong);
  border-color: var(--form-tile-border-active);
}

/* 图标按钮（次要行） */
.icon-btn {
  width: 32px;
  height: 32px;
  padding: 0;
  border: 1.5px solid var(--color-border-muted);
  background: white;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.icon-btn:hover {
  background: var(--color-surface-quiet);
  border-color: var(--form-tile-border-muted);
}

.icon-btn.edit-btn:hover {
  background: var(--form-tile-surface-muted);
  border-color: var(--form-tile-border-subtle);
}

.icon-btn.delete-btn:hover {
  background: var(--form-tile-surface-strong);
  border-color: var(--form-tile-border-active);
}
</style>
