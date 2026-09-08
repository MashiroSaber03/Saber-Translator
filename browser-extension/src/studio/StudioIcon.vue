<script setup lang="ts">
defineProps<{ name: 'translate' | 'settings' | 'tasks' | 'close' }>()
const paths = {
  translate: ['M4 4h16v12H9l-5 4V4Z', 'M8 8h8M8 12h5'],
  settings: ['M4 7h16M4 17h16', 'M9 4v6M15 14v6'],
  tasks: ['M9 6h11M9 12h11M9 18h11', 'm3 6 1 1 2-2m-3 7 1 1 2-2m-3 7 1 1 2-2'],
  close: ['m6 6 12 12M6 18 18 6'],
}
</script>
<template>
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.7"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <path v-for="(d, i) in paths[name]" :key="i" :d="d" />
  </svg>
</template>
