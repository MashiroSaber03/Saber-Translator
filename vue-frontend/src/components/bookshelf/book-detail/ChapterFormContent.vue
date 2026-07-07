<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiInput from '@/components/ui/UiInput.vue'

defineProps<{
  modelValue: string
}>()

defineEmits<{
  (event: 'update:modelValue', value: string): void
  (event: 'save'): void
}>()
</script>

<template>
  <UiField
    class="chapter-form-content__field"
    label="章节名称"
    control-id="chapterTitleInput"
    variant="dialog"
    required
  >
    <UiInput
      id="chapterTitleInput"
      :model-value="modelValue"
      type="text"
      autocomplete="off"
      placeholder="例如：第1话、序章"
      @update:model-value="$emit('update:modelValue', String($event))"
      @keydown.enter="$emit('save')"
    />
  </UiField>
</template>
