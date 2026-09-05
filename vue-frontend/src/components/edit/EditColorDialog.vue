<script setup lang="ts">
import { ref } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiColorPicker from '@/components/ui/UiColorPicker.vue'
import type { BubbleColorField } from '@/types/bubble'
import { BUBBLE_COLOR_LABELS } from './bubbleColorFields'

const props = defineProps<{ field: BubbleColorField; color: string }>()
const emit = defineEmits<{
  apply: [field: BubbleColorField, color: string]
  pick: [field: BubbleColorField]
  close: []
}>()
const draft = ref(props.color)
const valid = ref(true)
</script>

<template>
  <BaseModal :title="BUBBLE_COLOR_LABELS[field]" size="small" width="380px" @close="emit('close')">
    <UiColorPicker v-model="draft" @validity-change="valid = $event" />
    <template #footer>
      <UiButton variant="secondary" @click="emit('pick', field)">从图片取色</UiButton>
      <UiButton variant="secondary" @click="emit('close')">取消</UiButton>
      <UiButton variant="primary" :disabled="!valid" @click="emit('apply', field, draft)">应用颜色</UiButton>
    </template>
  </BaseModal>
</template>
