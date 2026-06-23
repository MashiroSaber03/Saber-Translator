<script setup lang="ts">
defineProps<{
  target: 'book' | 'chapter'
}>()
</script>

<template>
  <p>
    {{
      target === 'book'
        ? '确定要删除这本书籍吗？所有章节数据将一并删除，此操作不可恢复。'
        : '确定要删除这个章节吗？此操作不可恢复。'
    }}
  </p>
</template>
