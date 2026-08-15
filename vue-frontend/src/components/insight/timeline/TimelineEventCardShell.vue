<script setup lang="ts">
import { ref, watch } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'

const props = defineProps<{
  badgeLabel: string
  expanded: boolean
  pageRangeLabel: string
  thumbnailPage: number
  thumbnailUrl: string
  toggleAriaLabel: string
}>()

defineEmits<{
  (event: 'showPage', pageNum: number): void
  (event: 'toggle'): void
}>()

const thumbnailFailed = ref(false)

function handleThumbnailError(): void {
  thumbnailFailed.value = true
}

watch(
  () => [props.thumbnailUrl, props.thumbnailPage] as const,
  () => {
    thumbnailFailed.value = false
  }
)
</script>

<template>
  <div class="timeline-event-card-shell">
    <div class="timeline-event-card-shell__header">
      <UiButton
        variant="toolbar"
        type="button"
        class="timeline-event-card-shell__thumbnail-action"
        :aria-label="`查看第 ${thumbnailPage} 页`"
        @click="$emit('showPage', thumbnailPage)"
      >
        <img
          v-if="thumbnailUrl && !thumbnailFailed"
          class="timeline-event-card-shell__thumbnail"
          :src="thumbnailUrl"
          :alt="`第${thumbnailPage}页`"
          loading="lazy"
          @error="handleThumbnailError"
        />
        <span
          v-else
          class="timeline-event-card-shell__thumbnail timeline-event-card-shell__thumbnail-fallback"
        >
          第{{ thumbnailPage }}页
        </span>
      </UiButton>

      <UiButton
        variant="toolbar"
        type="button"
        class="timeline-event-card-shell__toggle"
        :aria-label="toggleAriaLabel"
        :aria-expanded="String(expanded)"
        @click="$emit('toggle')"
      >
        <span class="timeline-event-card-shell__title">
          <span class="timeline-event-card-shell__page-range">{{ pageRangeLabel }}</span>
          <span class="timeline-event-card-shell__badge">{{ badgeLabel }}</span>
        </span>
        <UiIcon
          name="chevron-right"
          size="14"
          class="timeline-event-card-shell__expand-icon"
          :class="{ 'timeline-event-card-shell__expand-icon--expanded': expanded }"
        />
      </UiButton>
    </div>

    <div
      v-if="$slots.summary"
      class="timeline-event-card-shell__summary"
      :class="{ 'timeline-event-card-shell__summary--expanded': expanded }"
    >
      <slot name="summary" />
    </div>

    <slot />
  </div>
</template>

<style scoped>
.timeline-event-card-shell {
  flex: 1;
  overflow: hidden;
  border: 1px solid var(--color-border-muted);
  border-radius: 12px;
  background: var(--insight-surface-secondary);
  transition:
    transform 0.2s,
    box-shadow 0.2s;
}

.timeline-event-card-shell:hover {
  box-shadow: 0 4px 12px var(--timeline-panel-card-shadow);
  transform: translateX(4px);
}

.timeline-event-card-shell__header {
  display: flex;
  gap: 12px;
  padding: 12px;
  border-bottom: 1px solid var(--color-border-muted);
  background: var(--insight-surface-tertiary);
}

.timeline-event-card-shell__thumbnail-action {
  flex-shrink: 0;
  border-radius: 6px;
}

.timeline-event-card-shell__toggle {
  display: flex;
  flex: 1;
  align-items: stretch;
  justify-content: space-between;
  min-width: 0;
  text-align: left;
}

.timeline-event-card-shell__thumbnail {
  display: block;
  width: 60px;
  height: 80px;
  border-radius: 6px;
  background: var(--insight-surface-page);
  object-fit: cover;
  transition: transform 0.2s;
}

.timeline-event-card-shell__thumbnail-fallback {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--insight-text-secondary);
  font-size: 12px;
  font-weight: 600;
}

.timeline-event-card-shell__thumbnail-action:hover .timeline-event-card-shell__thumbnail {
  transform: scale(1.05);
}

.timeline-event-card-shell__title {
  display: flex;
  flex: 1;
  flex-direction: column;
  justify-content: center;
  gap: 4px;
  min-width: 0;
}

.timeline-event-card-shell__page-range {
  color: var(--insight-text-primary);
  font-size: 15px;
  font-weight: 600;
}

.timeline-event-card-shell__badge {
  width: fit-content;
  max-width: 100%;
  padding: 2px 8px;
  overflow: hidden;
  border-radius: 10px;
  background: var(--insight-surface-page);
  color: var(--insight-text-secondary);
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.timeline-event-card-shell__expand-icon {
  align-self: center;
  color: var(--insight-text-secondary);
  transition: transform 0.2s;
}

.timeline-event-card-shell__expand-icon--expanded {
  transform: rotate(90deg);
}

.timeline-event-card-shell__summary {
  display: -webkit-box;
  padding: 12px;
  overflow: hidden;
  border-bottom: 1px solid var(--color-border-muted);
  color: var(--insight-text-secondary);
  font-size: 14px;
  line-height: 1.6;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.timeline-event-card-shell__summary--expanded {
  display: block;
  overflow: visible;
}
</style>
