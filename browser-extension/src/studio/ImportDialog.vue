<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import SelectControl from './SelectControl.vue'
import type {
  BrowserLibraryBook,
  BrowserSessionImportCommand,
  BrowserSessionImportResult,
} from '../types'
const props = defineProps<{
  title: string
  loadBooks: () => Promise<BrowserLibraryBook[]>
  submit: (command: BrowserSessionImportCommand) => Promise<BrowserSessionImportResult>
}>()
const emit = defineEmits<{ close: [] }>()
const dialog = ref<HTMLElement>()
const previousFocus = document.activeElement as HTMLElement | null
onMounted(() => dialog.value?.focus())
onBeforeUnmount(() => previousFocus?.focus())
function trap(event: KeyboardEvent) {
  if (event.key !== 'Tab') return
  const nodes = [
    ...dialog.value!.querySelectorAll<HTMLElement>(
      'button:not(:disabled),input:not(:disabled),[tabindex="0"]'
    ),
  ].filter(node => node.checkVisibility())
  const first = nodes[0],
    last = nodes.at(-1)
  if (!first) {
    event.preventDefault()
    return
  }
  if (
    event.shiftKey &&
    (document.activeElement === first || document.activeElement === dialog.value)
  ) {
    event.preventDefault()
    last?.focus()
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault()
    first.focus()
  }
}
const books = ref<BrowserLibraryBook[]>([])
const destination = ref('new')
const bookTitle = ref(props.title)
const chapterTitle = ref(props.title)
const bookId = ref('')
const busy = ref(true)
const submitting = ref(false)
const error = ref('')
const options = computed(() =>
  books.value.map(book => ({
    value: book.id,
    label: `${book.title} · ${book.chapterCount} 章`,
  }))
)
const valid = computed(
  () =>
    chapterTitle.value.trim() &&
    (destination.value === 'new' ? bookTitle.value.trim() : bookId.value)
)
onMounted(async () => {
  try {
    books.value = await props.loadBooks()
    bookId.value = books.value[0]?.id ?? ''
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    busy.value = false
    await nextTick()
    dialog.value?.querySelector<HTMLInputElement>('input')?.focus()
  }
})
async function save() {
  if (!valid.value || busy.value) return
  busy.value = true
  submitting.value = true
  try {
    await props.submit(
      destination.value === 'new'
        ? {
            destination: 'new',
            bookTitle: bookTitle.value.trim(),
            chapterTitle: chapterTitle.value.trim(),
          }
        : {
            destination: 'existing',
            targetBookId: bookId.value,
            chapterTitle: chapterTitle.value.trim(),
          }
    )
    emit('close')
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    busy.value = false
    submitting.value = false
  }
}
</script>
<template>
  <div class="dialog-backdrop" @keydown.esc.stop="!submitting && $emit('close')">
    <section
      ref="dialog"
      class="dialog"
      tabindex="-1"
      @keydown="trap"
      role="dialog"
      aria-modal="true"
      aria-label="导入到书架"
    >
      <div class="view-heading">
        <div>
          <h2>留在你的书架</h2>
          <p>保存图片，下一次继续阅读</p>
        </div>
        <button
          class="icon-button"
          aria-label="关闭导入"
          :disabled="submitting"
          @click="$emit('close')"
        >
          ×
        </button>
      </div>
      <div v-if="error" class="notice error" role="status">{{ error }}</div>
      <form @submit.prevent="save">
        <fieldset :disabled="busy">
          <div class="segmented">
            <button
              type="button"
              :class="{ active: destination === 'new' }"
              @click="destination = 'new'"
            >
              新建书籍</button
            ><button
              type="button"
              :class="{ active: destination === 'existing' }"
              :disabled="!books.length"
              @click="destination = 'existing'"
            >
              已有书籍
            </button>
          </div>
          <label v-if="destination === 'new'" class="field"
            >书籍名称<input v-model="bookTitle" required maxlength="500" /></label
          ><label v-else class="field"
            >选择书籍<SelectControl
              v-model="bookId"
              :options="options"
              label="选择书籍"
              searchable /></label
          ><label class="field"
            >章节名称<input v-model="chapterTitle" required maxlength="500" /></label
          ><button class="button primary full" type="submit" :disabled="!valid || busy">
            {{ busy ? '正在导入…' : '确认导入' }}
          </button>
        </fieldset>
      </form>
    </section>
  </div>
</template>
