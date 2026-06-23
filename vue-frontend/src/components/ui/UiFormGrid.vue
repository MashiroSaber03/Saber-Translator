<template>
  <div class="ui-form-grid">
    <slot />
  </div>
</template>

<style scoped>
.ui-form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 15px;
  margin-bottom: 15px;
}

.ui-form-grid :slotted(.ui-settings-field) {
  margin-bottom: 0;
}
</style>
