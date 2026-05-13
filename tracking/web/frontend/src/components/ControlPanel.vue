<template>
  <div class="control-panel">
    <div class="panel-header">Controls</div>

    <div class="control-panel__buttons">
      <button
        @click="onPause"
        :class="['btn', isPaused ? 'btn-success' : '']"
        :disabled="!isRunning"
      >
        <svg v-if="!isPaused" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <rect x="6" y="4" width="4" height="16"/>
          <rect x="14" y="4" width="4" height="16"/>
        </svg>
        <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <polygon points="5,3 19,12 5,21"/>
        </svg>
        {{ isPaused ? 'Resume' : 'Pause' }}
      </button>

      <button
        @click="onQuit"
        class="btn btn-danger"
        :disabled="!isRunning"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M18 6L6 18M6 6l12 12"/>
        </svg>
        Quit
      </button>
    </div>

    <div class="control-panel__status">
      <div class="control-panel__status-item">
        <span class="text-muted">Status:</span>
        <span v-if="!isRunning" class="badge badge-warning">Stopped</span>
        <span v-else-if="isPaused" class="badge badge-info">Paused</span>
        <span v-else class="badge badge-success">Running</span>
      </div>

      <div v-if="currentTarget" class="control-panel__status-item">
        <span class="text-muted">Target:</span>
        <span class="control-panel__target-text">{{ currentTarget }}</span>
      </div>

      <div v-if="isRunning" class="control-panel__status-item">
        <span class="text-muted">Frame:</span>
        <span>{{ currentFrame }} / {{ totalFrames }}</span>
      </div>

      <div v-if="isRunning" class="control-panel__status-item">
        <span class="text-muted">FPS:</span>
        <span>{{ fps?.toFixed(1) || '0.0' }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  isRunning: {
    type: Boolean,
    default: false
  },
  isPaused: {
    type: Boolean,
    default: false
  },
  currentTarget: {
    type: String,
    default: ''
  },
  currentFrame: {
    type: Number,
    default: 0
  },
  totalFrames: {
    type: Number,
    default: 0
  },
  fps: {
    type: Number,
    default: 0
  }
})

const emit = defineEmits(['pause', 'quit'])

function onPause() {
  emit('pause')
}

function onQuit() {
  if (confirm('Are you sure you want to quit tracking?')) {
    emit('quit')
  }
}
</script>

<style scoped>
.control-panel {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.control-panel__buttons {
  display: flex;
  gap: var(--spacing-sm);
}

.control-panel__buttons .btn {
  flex: 1;
}

.control-panel__status {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  padding-top: var(--spacing-md);
  border-top: 1px solid var(--color-border);
}

.control-panel__status-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: var(--font-size-sm);
}

.control-panel__target-text {
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
