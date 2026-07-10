<script setup lang="ts">
import UiIcon from '@/components/ui/UiIcon.vue'
import type { UiIconName } from '@/components/ui/iconRegistry'

withDefaults(defineProps<{
  appearance?: 'default' | 'reading'
  ariaLabel?: string
  avatarIconName?: UiIconName
  avatarImageSrc?: string
  avatarLabel: string
  role: 'assistant' | 'user'
}>(), {
  appearance: 'default',
  ariaLabel: undefined,
  avatarIconName: undefined,
  avatarImageSrc: undefined,
})
</script>

<template>
  <article
    class="product-message-bubble"
    :class="[
      `product-message-bubble--${role}`,
      `product-message-bubble--appearance-${appearance}`,
    ]"
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

.product-message-bubble--appearance-reading {
  gap: 0;
}

.product-message-bubble--appearance-reading .product-message-bubble__avatar {
  display: none;
}

.product-message-bubble--appearance-reading .product-message-bubble__body {
  display: grid;
  grid-template-areas:
    "meta actions"
    "content content"
    "footer footer";
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 0 12px;
  width: min(100%, var(--product-message-bubble-reading-width, 88%));
  max-width: none;
  padding: var(--product-message-bubble-reading-padding, 14px 16px);
  border-color: var(--product-message-bubble-reading-border, var(--color-border-muted));
  border-radius: var(--product-message-bubble-reading-radius, 18px);
  background: var(--product-message-bubble-reading-assistant-background, var(--color-surface-card));
  box-shadow: var(--product-message-bubble-reading-shadow, none);
  color: var(--product-message-bubble-reading-text, var(--color-text-default));
}

.product-message-bubble--appearance-reading.product-message-bubble--assistant .product-message-bubble__body {
  margin-right: auto;
  border-bottom-left-radius: var(--product-message-bubble-reading-radius, 18px);
}

.product-message-bubble--appearance-reading.product-message-bubble--user .product-message-bubble__body {
  margin-left: auto;
  border-color: var(--product-message-bubble-reading-user-border, var(--product-message-bubble-reading-border, var(--color-border-muted)));
  border-bottom-right-radius: var(--product-message-bubble-reading-radius, 18px);
  background: var(--product-message-bubble-reading-user-background, var(--color-surface-muted));
  color: var(--product-message-bubble-reading-text, var(--color-text-default));
}

.product-message-bubble--appearance-reading .product-message-bubble__meta {
  grid-area: meta;
  margin-bottom: 10px;
}

.product-message-bubble--appearance-reading .product-message-bubble__content {
  grid-area: content;
}

.product-message-bubble--appearance-reading .product-message-bubble__footer {
  grid-area: footer;
}

.product-message-bubble--appearance-reading .product-message-bubble__actions {
  grid-area: actions;
  align-self: start;
  justify-content: flex-end;
  margin-top: 0;
}

@media (--breakpoint-sm-down) {
  .product-message-bubble__body {
    max-width: calc(100% - 48px);
  }

  .product-message-bubble--appearance-reading .product-message-bubble__body {
    grid-template-areas:
      "meta"
      "content"
      "footer"
      "actions";
    grid-template-columns: 1fr;
    width: 100%;
    max-width: none;
  }

  .product-message-bubble--appearance-reading .product-message-bubble__actions {
    justify-content: flex-start;
    margin-top: 12px;
  }
}
</style>
