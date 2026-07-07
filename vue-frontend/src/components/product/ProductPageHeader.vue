<script setup lang="ts">
type ProductPageHeaderVariant = 'default' | 'brand' | 'fixed' | 'reader'

withDefaults(defineProps<{
  actionsLabel?: string
  variant?: ProductPageHeaderVariant
  navLabel?: string
  logoTitle?: string
  homeTo?: string
}>(), {
  actionsLabel: '页面操作',
  variant: 'default',
  navLabel: '页面导航',
  logoTitle: '返回书架',
  homeTo: '/',
})
</script>

<template>
  <header class="product-page-header" :class="`product-page-header--${variant}`">
    <div class="product-page-header__content">
      <div class="product-page-header__brand">
        <slot v-if="$slots.brand" name="brand" />
        <RouterLink v-else :to="homeTo" :title="logoTitle" class="product-page-header__brand-link">
          <img src="/pic/logo.png" alt="Saber-Translator Logo" class="product-page-header__logo">
          <span class="product-page-header__name">Saber-Translator</span>
        </RouterLink>
      </div>

      <div v-if="$slots.meta" class="product-page-header__meta">
        <slot name="meta" />
      </div>

      <nav v-if="$slots.nav" class="product-page-header__nav" :aria-label="navLabel">
        <slot name="nav" />
      </nav>

      <div
        v-if="$slots.actions"
        class="product-page-header__actions"
        role="group"
        :aria-label="actionsLabel"
      >
        <slot name="actions" />
      </div>
    </div>
  </header>
</template>

<style scoped>
.product-page-header {
  --product-page-header-background: var(--color-surface-raised);
  --product-page-header-shadow: var(--shadow-soft);
  --product-page-header-brand-text: var(--color-text-heading);
  --product-page-header-content-padding: 6px 10px;
  --product-page-header-content-radius: 12px;
  --product-page-header-logo-size: 40px;
  --product-page-header-logo-shadow: none;
  --product-page-header-gap: 15px;
  --product-header-meta-pill-background: var(--color-surface-muted);
  --product-header-meta-pill-text: var(--color-text-default);

  position: relative;
  z-index: var(--z-app-header);
  display: flex;
  align-items: center;
  justify-content: center;
  width: min(980px, calc(100% - 40px));
  max-width: none;
  margin: 0 auto;
  padding: 10px 20px;
  color: var(--color-text-heading);
  background: transparent;
}

.product-page-header__content {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  gap: var(--product-page-header-gap);
  padding: var(--product-page-header-content-padding);
  background: var(--product-page-header-background);
  border-radius: var(--product-page-header-content-radius);
  box-shadow: 0 2px 10px var(--product-page-header-shadow);
}

.product-page-header__brand {
  position: relative;
  z-index: var(--z-local);
  display: flex;
  flex: 0 1 auto;
  align-items: center;
  min-width: 0;
}

.product-page-header__brand-link {
  display: flex;
  align-items: center;
  gap: 15px;
  min-width: 0;
  color: var(--product-page-header-brand-text);
  text-decoration: none;
}

.product-page-header__logo {
  width: var(--product-page-header-logo-size);
  height: var(--product-page-header-logo-size);
  border-radius: 8px;
  box-shadow: var(--product-page-header-logo-shadow);
  object-fit: cover;
}

.product-page-header__name {
  min-width: 0;
  font-size: 1.5em;
  font-weight: 700;
  letter-spacing: 0;
  line-height: 1.35;
  white-space: nowrap;
}

.product-page-header__meta,
.product-page-header__nav,
.product-page-header__actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  min-width: 0;
  gap: var(--product-page-header-gap);
}

.product-page-header__nav {
  flex: 1 1 auto;
  justify-content: flex-end;
}

.product-page-header__actions {
  flex: 0 0 auto;
}

.product-page-header--brand {
  --product-page-header-background: transparent;
  --product-page-header-brand-text: var(--color-text-inverse);
  --product-page-header-content-padding: 0;
  --product-page-header-content-radius: 0;
  --product-page-header-shadow: transparent;
  --product-page-header-logo-size: 40px;
  --product-page-header-logo-shadow: 0 2px 8px var(--shadow-medium);
  --product-page-header-gap: 16px;
  --product-header-action-context-surface: var(--color-overlay-inverse-soft);
  --product-header-action-context-border: var(--color-overlay-inverse-raised);
  --product-header-action-context-text: var(--color-text-inverse);
  --product-header-action-context-hover-surface: var(--color-overlay-inverse-emphasis);
  --product-header-action-context-hover-border: var(--color-overlay-inverse-emphasis);
  --product-header-action-context-solid-surface: var(--color-overlay-inverse-muted);
  --product-header-action-context-solid-hover-surface: var(--color-overlay-inverse-emphasis);
  --product-header-action-context-solid-shadow: var(--color-overlay-inverse-raised);
  --product-header-action-context-plain-text: var(--color-text-inverse);
  --product-header-meta-pill-background: var(--color-overlay-inverse-raised);
  --product-header-meta-pill-text: var(--color-text-inverse);

  position: sticky;
  top: 0;
  width: 100%;
  max-width: none;
  min-height: 64px;
  margin: 0;
  padding: 8px 24px;
  background: linear-gradient(135deg, var(--color-action-brand) 0%, var(--color-action-brand-strong) 100%);
  box-shadow: 0 2px 20px var(--shadow-action-brand);
}

.product-page-header--brand .product-page-header__content {
  width: 100%;
  max-width: 1400px;
  margin: 0 auto;
}

.product-page-header--brand .product-page-header__brand-link {
  gap: 12px;
}

.product-page-header--brand .product-page-header__name {
  font-size: 1.3rem;
  white-space: nowrap;
}

.product-page-header--fixed {
  --product-page-header-background: transparent;
  --product-page-header-brand-text: var(--color-text-default);
  --product-page-header-content-padding: 0;
  --product-page-header-content-radius: 0;
  --product-page-header-shadow: transparent;
  --product-page-header-logo-size: 32px;
  --product-header-action-context-surface: transparent;
  --product-header-action-context-hover-surface: var(--color-surface-interactive-hover);

  position: fixed;
  top: 0;
  right: 0;
  left: 0;
  width: auto;
  max-width: none;
  height: 56px;
  margin: 0;
  padding: 0 20px;
  background: var(--color-surface-subtle);
  border-bottom: 1px solid var(--color-border-muted);
}

.product-page-header--fixed .product-page-header__content {
  flex-wrap: nowrap;
  gap: 0;
  height: 100%;
  overflow: hidden;
}

.product-page-header--fixed .product-page-header__brand-link {
  gap: 10px;
}

.product-page-header--fixed .product-page-header__brand,
.product-page-header--fixed .product-page-header__brand-link,
.product-page-header--fixed .product-page-header__nav,
.product-page-header--fixed .product-page-header__actions {
  flex: 0 1 auto;
  min-width: auto;
}

.product-page-header--fixed .product-page-header__nav,
.product-page-header--fixed .product-page-header__actions {
  justify-content: flex-start;
  gap: 16px;
}

.product-page-header--fixed .product-page-header__name {
  font-size: 18px;
  font-weight: 600;
}

.product-page-header--reader {
  --product-page-header-background: transparent;
  --product-page-header-brand-text: var(--color-text-inverse);
  --product-page-header-content-padding: 0;
  --product-page-header-content-radius: 0;
  --product-page-header-shadow: transparent;
  --product-page-header-reader-shadow: var(--shadow-medium);
  --product-page-header-gap: 12px;
  --product-header-action-context-surface: var(--color-overlay-inverse-soft);
  --product-header-action-context-border: var(--color-overlay-inverse-raised);
  --product-header-action-context-text: var(--color-text-inverse);
  --product-header-action-context-hover-surface: var(--color-overlay-inverse-emphasis);
  --product-header-action-context-hover-border: var(--color-overlay-inverse-emphasis);
  --product-header-action-context-solid-surface: var(--color-surface-base);
  --product-header-action-context-solid-hover-surface: var(--color-surface-base);
  --product-header-action-context-solid-shadow: var(--shadow-medium);
  --product-header-action-context-solid-text: var(--color-action-brand);
  --product-header-action-context-plain-text: var(--color-text-inverse);
  --product-header-action-context-active-surface: var(--color-surface-base);
  --product-header-action-context-active-text: var(--color-action-brand);
  --product-header-meta-pill-background: var(--color-overlay-inverse-raised);
  --product-header-meta-pill-text: var(--color-text-inverse);

  width: 100%;
  max-width: none;
  height: 56px;
  margin: 0;
  padding: 0 16px;
  color: var(--color-text-inverse);
  background: linear-gradient(135deg, var(--color-action-brand) 0%, var(--color-action-brand-strong) 100%);
  box-shadow: 0 2px 10px var(--product-page-header-reader-shadow);
}

.product-page-header--reader .product-page-header__content {
  position: relative;
  flex-wrap: nowrap;
  width: 100%;
  gap: var(--product-page-header-gap);
}

.product-page-header--reader .product-page-header__brand,
.product-page-header--reader .product-page-header__actions {
  flex: 0 1 auto;
}

.product-page-header--reader .product-page-header__meta {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
}

.product-page-header--reader .product-page-header__actions {
  justify-content: flex-end;
}

@media (--breakpoint-md-down) {
  .product-page-header--default {
    width: 100%;
    max-width: none;
    padding: 8px 10px;
  }

  .product-page-header--default .product-page-header__content {
    flex-wrap: wrap;
    justify-content: center;
    width: auto;
    gap: 10px 8px;
    background: transparent;
    box-shadow: none;
    padding: 9px 0 1px;
  }

  .product-page-header--default .product-page-header__brand {
    flex: 1 1 100%;
    justify-content: center;
  }

  .product-page-header--default .product-page-header__logo {
    width: 30px;
    height: 30px;
  }

  .product-page-header--default .product-page-header__name {
    display: none;
  }

  .product-page-header--default .product-page-header__nav,
  .product-page-header--default .product-page-header__actions {
    flex: 0 0 auto;
    flex-wrap: wrap;
    justify-content: center;
    width: 100%;
    max-width: 360px;
    gap: 10px 8px;
  }

  .product-page-header--fixed {
    padding: 0 12px;
  }

  .product-page-header--fixed .product-page-header__name {
    display: none;
    line-height: 1.35;
  }

  .product-page-header--fixed .product-page-header__nav,
  .product-page-header--fixed .product-page-header__actions {
    gap: 8px;
  }
}
</style>
