<script setup lang="ts">
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'

withDefaults(defineProps<{
  compact?: boolean
  dismissible?: boolean
}>(), {
  compact: false,
  dismissible: false,
})

const emit = defineEmits<{
  dismiss: []
}>()
</script>

<template>
  <ProductStatusBanner
    class="public-trial-notice"
    :class="{ 'public-trial-notice--compact': compact }"
    icon-name="globe"
    role="note"
    title="Saber Translator 在线试用版"
    tone="neutral"
  >
    <div class="public-trial-notice__copy">
      <p>本网站用于在线体验 Saber Translator 的漫画翻译与分析功能。</p>
      <p>
        共享服务器在高峰期可能需要等待，部分高性能模型或功能可能暂时关闭，试用数据会不定期清理。
      </p>
      <p class="public-trial-notice__personal-edition">
        如需完整、流畅且数据保存在自己电脑上的体验，请下载个人版。
        <strong>个人版完全开源免费，不包含任何收费功能。</strong>
      </p>
    </div>
    <nav class="public-trial-notice__links" aria-label="个人版项目链接">
      <a
        class="public-trial-notice__link public-trial-notice__link--primary"
        href="https://www.mashirosaber.top/"
        target="_blank"
        rel="noopener noreferrer"
      >
        <UiIcon name="globe" size="14" aria-hidden="true" />
        前往官网
      </a>
      <a
        class="public-trial-notice__link"
        href="https://github.com/MashiroSaber03/Saber-Translator"
        target="_blank"
        rel="noopener noreferrer"
      >
        <UiIcon name="github" size="14" aria-hidden="true" />
        查看 GitHub
      </a>
    </nav>
    <template v-if="dismissible" #actions>
      <UiIconButton
        label="关闭试用说明"
        title="关闭试用说明"
        variant="plain"
        size="sm"
        shape="circle"
        @click="emit('dismiss')"
      >
        <UiIcon name="x" size="15" />
      </UiIconButton>
    </template>
  </ProductStatusBanner>
</template>

<style scoped>
.public-trial-notice {
  --product-status-banner-padding: 14px 16px;
  --product-status-banner-radius: 12px;
  --product-status-banner-border: 1px solid var(--color-border-muted);
  --product-status-banner-background: var(--color-surface-card);
  --product-status-banner-actions-margin-left: auto;

  box-shadow: 0 8px 24px var(--shadow-soft);
}

.public-trial-notice__copy {
  display: grid;
  gap: 5px;
}

.public-trial-notice__copy p {
  margin: 0;
}

.public-trial-notice__personal-edition {
  color: var(--color-text-supporting);
}

.public-trial-notice__personal-edition strong {
  color: var(--color-text-strong);
  font-weight: 650;
}

.public-trial-notice__links {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.public-trial-notice__link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-height: 32px;
  padding: 6px 11px;
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  background: var(--color-surface-card);
  color: var(--color-text-default);
  font-size: 0.82rem;
  font-weight: 600;
  line-height: 1;
  text-decoration: none;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}

.public-trial-notice__link:hover {
  border-color: var(--color-text-supporting);
  background: var(--color-surface-interactive-hover);
  color: var(--color-text-strong);
}

.public-trial-notice__link:focus-visible {
  outline: 2px solid var(--color-border-brand);
  outline-offset: 2px;
}

.public-trial-notice__link--primary {
  border-color: var(--color-text-strong);
  background: var(--color-text-strong);
  color: var(--color-surface-base);
}

.public-trial-notice__link--primary:hover {
  border-color: var(--color-text-default);
  background: var(--color-text-default);
  color: var(--color-surface-base);
}

.public-trial-notice--compact {
  --product-status-banner-padding: 12px;
  --product-status-banner-gap: 9px;
  --product-status-banner-body-font-size: 0.84rem;

  box-shadow: none;
}

.public-trial-notice--compact .public-trial-notice__copy {
  gap: 4px;
}

.public-trial-notice--compact .public-trial-notice__links {
  margin-top: 10px;
}

.public-trial-notice--compact .public-trial-notice__link {
  min-height: 29px;
  padding: 5px 9px;
  font-size: 0.78rem;
}

@media (--breakpoint-preview-down) {
  .public-trial-notice__links {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
