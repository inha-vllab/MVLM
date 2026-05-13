<template>
  <div class="status-bar">
    <div class="status-bar__section">
      <div v-if="status?.is_running" class="status-bar__item">
        <span class="status-bar__indicator status-bar__indicator--running"></span>
        <span>Tracking</span>
      </div>
      <div v-else class="status-bar__item">
        <span class="status-bar__indicator status-bar__indicator--stopped"></span>
        <span>Stopped</span>
      </div>
    </div>

    <div class="status-bar__section">
      <div v-if="status?.current_frame !== undefined" class="status-bar__item">
        <span class="text-muted">Frame:</span>
        <span>{{ status.current_frame }} / {{ status.total_frames || '?' }}</span>
      </div>

      <div v-if="status?.fps !== undefined" class="status-bar__item">
        <span class="text-muted">FPS:</span>
        <span :class="fpsClass">{{ status.fps?.toFixed(1) || '0.0' }}</span>
      </div>

    </div>

    <div class="status-bar__section">
      <div v-if="status?.config" class="status-bar__item status-bar__item--compact">
        <span class="text-muted">{{ status.config }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  status: {
    type: Object,
    default: null
  }
})

const fpsClass = computed(() => {
  const fps = props.status?.fps || 0
  if (fps < 10) return 'text-danger'
  if (fps < 20) return 'text-warning'
  return 'text-success'
})
</script>

<style scoped>
.status-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-bg-secondary);
  border-top: 1px solid var(--color-border);
  font-size: var(--font-size-xs);
}

.status-bar__section {
  display: flex;
  gap: var(--spacing-md);
  align-items: center;
}

.status-bar__item {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
}

.status-bar__item--compact {
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.status-bar__indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-bar__indicator--running {
  background: #3fb950;
  animation: pulse 1.5s infinite;
}

.status-bar__indicator--stopped {
  background: var(--color-text-muted);
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.text-success { color: #3fb950; }
.text-warning { color: #d29922; }
.text-danger { color: #f85149; }
</style>
