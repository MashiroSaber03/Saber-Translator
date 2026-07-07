<script setup lang="ts">
import UiIcon from '@/components/ui/UiIcon.vue'
import type { UiIconName } from '@/components/ui/iconRegistry'

withDefaults(defineProps<{
  ariaLabel?: string
  avatarIconName?: UiIconName
  avatarImageSrc?: string
  avatarLabel: string
  role: 'assistant' | 'user'
}>(), {
  ariaLabel: undefined,
  avatarIconName: undefined,
  avatarImageSrc: undefined,
})
</script>

<template>
  <article
    class="product-message-bubble"
    :class="`product-message-bubble--${role}`"
    :aria-label="ariaLabel"
  >
    <div
      class="product-message-bubble__avatar"
      role="img"
      :aria-label="avatarLabel"
    >
      <img
        v-if="avatarImageSrc"
        class="product-message-bubble__avatar-image"
        :src="avatarImageSrc"
        alt=""
      >
      <UiIcon v-else-if="avatarIconName" :name="avatarIconName" />
      <slot v-else name="avatar" />
    </div>

    <div class="product-message-bubble__body">
      <div v-if="$slots.meta" class="product-message-bubble__meta">
        <slot name="meta" />
      </div>
      <div class="product-message-bubble__content">
        <slot />
      </div>
      <div v-if="$slots.footer" class="product-message-bubble__footer">
        <slot name="footer" />
      </div>
      <div v-if="$slots.actions" class="product-message-bubble__actions">
        <slot name="actions" />
      </div>
    </div>
  </article>
</template>

<style scoped>
.product-message-bubble {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  animation: slideIn 0.3s ease;
}

.product-message-bubble--user {
  flex-direction: row-reverse;
}

.product-message-bubble__avatar {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  overflow: hidden;
  border-radius: 50%;
  background: var(--color-surface-muted);
  color: var(--color-text-supporting);
  font-size: 18px;
}

.product-message-bubble--user .product-message-bubble__avatar {
  background: transparent;
}

.product-message-bubble__avatar-image {
  display: block;
  width: 100%;
  height: 100%;
  border-radius: inherit;
  object-fit: cover;
}

.product-message-bubble__body {
  max-width: min(70%, 760px);
  padding: 12px 16px;
  border: 1px solid var(--color-border-muted);
  border-radius: 12px;
  background: var(--color-surface-card);
  color: var(--color-text-default);
  line-height: 1.6;
}

.product-message-bubble--user .product-message-bubble__body {
  border-color: transparent;
  border-bottom-right-radius: 4px;
  background: var(--color-action-primary);
  color: var(--color-text-inverse);
}

.product-message-bubble--assistant .product-message-bubble__body {
  border-bottom-left-radius: 4px;
}

.product-message-bubble__content {
  min-width: 0;
  overflow-wrap: anywhere;
}

.product-message-bubble__meta {
  margin-bottom: 8px;
}

.product-message-bubble__footer {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--color-border-muted);
}

.product-message-bubble__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

@media (--breakpoint-sm-down) {
  .product-message-bubble__body {
    max-width: calc(100% - 48px);
  }
}
</style>
