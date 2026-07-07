<template>
  <header class="studio-topbar">
    <div class="studio-topbar__left">
      <ProductHeaderAction class="studio-topbar__action" icon-name="chevron-left" label="返回分析" @click="$emit('back')" />
      <ProductHeaderAction class="studio-topbar__action" icon-name="users" label="角色资源" @click="$emit('open-resource')" />
      <div class="studio-topbar__title-block">
        <div class="studio-topbar__title-row">
          <h1 class="studio-topbar__title">角色工坊 2.0</h1>
          <span
            v-if="busy && busyLabel"
            class="studio-topbar__status-pill studio-topbar__status-pill--busy"
          >
            {{ busyLabel }}
          </span>
        </div>
        <div class="studio-topbar__meta-row">
          <span v-if="bookTitle" class="studio-topbar__status-pill">当前书籍：{{ bookTitle }}</span>
          <span
            class="studio-topbar__status-pill"
            :class="{ 'studio-topbar__status-pill--empty': !documentTitle }"
          >
            {{ documentTitle ? `当前角色：${documentTitle}` : '当前角色：未选择' }}
          </span>
          <span v-if="documentOrigin" class="studio-topbar__status-pill">{{ documentOrigin }}</span>
        </div>
      </div>
    </div>

    <div class="studio-topbar__right">
      <ProductHeaderAction class="studio-topbar__action" icon-name="download" label="导出区" @click="$emit('open-export')" />
      <ProductHeaderAction
        class="studio-topbar__action"
        icon-name="target"
        :disabled="!hasDocument || validatePending"
        :label="validatePending ? '诊断中...' : '诊断'"
        @click="$emit('validate')"
      />
      <ProductHeaderAction
        variant="solid"
        class="studio-topbar__action studio-topbar__action--primary"
        icon-name="save"
        :disabled="!hasDocument || savePending"
        :label="savePending ? '保存中...' : '保存'"
        @click="$emit('save')"
      />
    </div>
  </header>
</template>

<script setup lang="ts">
import ProductHeaderAction from '@/components/product/ProductHeaderAction.vue'
defineProps<{
  bookTitle: string
  documentTitle: string
  documentOrigin: string
  hasDocument: boolean
  busy: boolean
  busyLabel: string
  savePending: boolean
  validatePending: boolean
}>()

defineEmits<{
  (e: 'back'): void
  (e: 'save'): void
  (e: 'validate'): void
  (e: 'open-resource'): void
  (e: 'open-export'): void
}>()
</script>

<style scoped>
.studio-topbar {
  --studio-topbar-backdrop-background: color-mix(in srgb, var(--color-surface-card) 90%, transparent);
  --studio-topbar-primary-action-end: var(--color-action-brand-strong);
  --studio-topbar-primary-action-shadow: var(--shadow-action-brand);
  --studio-topbar-primary-action-start: var(--color-action-brand);
  --studio-topbar-status-background: color-mix(in srgb, var(--color-action-brand) 6%, transparent);
  --studio-topbar-title-text: var(--color-text-heading);
  --product-header-action-context-surface: var(--studio-surface-muted);
  --product-header-action-context-text: var(--studio-text-default);
  --product-header-action-context-hover-surface: var(--studio-surface-tint-muted);
  --product-header-action-context-solid-surface: linear-gradient(135deg, var(--studio-topbar-primary-action-start), var(--studio-topbar-primary-action-end));
  --product-header-action-context-solid-shadow: var(--studio-topbar-primary-action-shadow);
  --product-header-action-context-solid-text: var(--color-text-inverse);

  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
  padding: 10px 20px;
  position: sticky;
  top: 0;
  z-index: var(--z-app-header);
  border-bottom: 1px solid var(--studio-border-default);
  background: var(--studio-topbar-backdrop-background);
  backdrop-filter: blur(18px);
}

.studio-topbar__left,
.studio-topbar__right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.studio-topbar__left {
  min-width: 0;
  flex: 1;
}

.studio-topbar__right {
  flex-shrink: 0;
  justify-content: flex-end;
  flex-wrap: wrap;
}

.studio-topbar__title-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
  flex: 0 1 420px;
  max-width: 420px;
  padding: 8px 12px;
  border-radius: 18px;
  background: var(--color-surface-raised);
  border: 1px solid var(--studio-border-default);
}

.studio-topbar__title-row {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.studio-topbar__title {
  margin: 0;
  font-size: 20px;
  line-height: 1.1;
  color: var(--studio-topbar-title-text);
  white-space: nowrap;
}

.studio-topbar__meta-row {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-items: center;
}

.studio-topbar__status-pill {
  border-radius: 999px;
  padding: 4px 9px;
  background: var(--studio-topbar-status-background);
  color: var(--studio-text-default);
  font-size: 11px;
  line-height: 1.2;
}

.studio-topbar__status-pill--empty {
  color: var(--studio-text-subtle);
}

.studio-topbar__status-pill--busy {
  background: var(--studio-surface-tint-muted);
  color: var(--color-text-primary-strong);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 160px;
  flex-shrink: 1;
}

.studio-topbar__action {
  border-radius: 14px;
  font-size: 14px;
}

.studio-topbar__action--primary {
  padding-inline: 18px;
}

@media (--breakpoint-lg-down) {
  .studio-topbar {
    flex-wrap: wrap;
    padding: 12px 16px;
  }

  .studio-topbar__left,
  .studio-topbar__right {
    width: 100%;
    min-width: 0;
    flex-wrap: wrap;
  }

  .studio-topbar__right {
    justify-content: flex-start;
  }

  .studio-topbar__title-block {
    flex: 1 1 100%;
    max-width: none;
  }

  .studio-topbar__title-row {
    flex-wrap: wrap;
  }

  .studio-topbar__status-pill--busy {
    max-width: none;
  }
}
</style>
