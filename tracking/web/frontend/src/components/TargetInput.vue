<template>
  <div class="target-input">
    <div class="panel-header">Switch Target</div>

    <form @submit.prevent="onSubmit" class="target-input__form">
      <div class="form-group">
        <label class="label">Description</label>
        <input
          ref="textInput"
          v-model="text"
          type="text"
          class="input"
          placeholder="e.g., the woman in red dress"
          :disabled="disabled"
        />
      </div>

      <div v-if="!bboxDisabled" class="form-group">
        <div class="flex items-center gap-sm">
          <input
            v-model="useBbox"
            type="checkbox"
            id="use-bbox"
            class="checkbox"
            :disabled="disabled"
          />
          <label for="use-bbox" class="label" style="margin: 0; cursor: pointer;">
            Specify bounding box
          </label>
        </div>
      </div>

      <div v-if="useBbox && !bboxDisabled" class="form-group">
        <!-- Bbox input mode toggle -->
        <div class="target-input__bbox-mode">
          <button
            type="button"
            :class="['target-input__mode-btn', { 'target-input__mode-btn--active': bboxMode === 'draw' }]"
            :disabled="disabled"
            @click="setBboxMode('draw')"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" stroke-dasharray="4 2"/>
            </svg>
            Draw on video
          </button>
          <button
            type="button"
            :class="['target-input__mode-btn', { 'target-input__mode-btn--active': bboxMode === 'input' }]"
            :disabled="disabled"
            @click="setBboxMode('input')"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="4 7 4 4 20 4 20 7"/>
              <line x1="9" y1="20" x2="15" y2="20"/>
              <line x1="12" y1="4" x2="12" y2="20"/>
            </svg>
            Input values
          </button>
        </div>

        <!-- Draw mode -->
        <div v-if="bboxMode === 'draw'" class="target-input__draw-info">
          <div v-if="drawnBbox" class="target-input__drawn-bbox">
            <span class="text-muted">Selected:</span>
            <span>x={{ drawnBbox[0] }}, y={{ drawnBbox[1] }}, w={{ drawnBbox[2] }}, h={{ drawnBbox[3] }}</span>
          </div>
          <p v-else class="text-muted" style="font-size: 12px;">
            Drag on the video to select a bounding box
          </p>
        </div>

        <!-- Input mode -->
        <div v-else class="form-group">
          <label class="label">Bounding Box — x, y, w, h</label>
          <div class="target-input__bbox-inputs">
            <input
              v-model.number="bbox.x"
              type="number"
              class="input"
              placeholder="x"
              :disabled="disabled"
            />
            <input
              v-model.number="bbox.y"
              type="number"
              class="input"
              placeholder="y"
              :disabled="disabled"
            />
            <input
              v-model.number="bbox.w"
              type="number"
              class="input"
              placeholder="w"
              :disabled="disabled"
            />
            <input
              v-model.number="bbox.h"
              type="number"
              class="input"
              placeholder="h"
              :disabled="disabled"
            />
          </div>
        </div>
      </div>

      <button
        type="submit"
        class="btn btn-primary w-full"
        :disabled="disabled || !text.trim()"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M23 4v6h-6M1 20v-6h6"/>
          <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"/>
        </svg>
        Switch Target
      </button>

      <div v-if="message" :class="['status-message', message.type]">
        {{ message.text }}
      </div>
    </form>
  </div>
</template>

<script setup>
import { ref, reactive, watch } from 'vue'

const props = defineProps({
  disabled: {
    type: Boolean,
    default: false
  },
  selectedBbox: {
    type: Array,
    default: null
  },
  bboxDisabled: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['submit', 'drawing-toggle'])

const textInput = ref(null)
const text = ref('')
const useBbox = ref(false)
const bboxMode = ref('draw')  // 'draw' or 'input'
const drawnBbox = ref(null)
const bbox = reactive({
  x: null,
  y: null,
  w: null,
  h: null
})
const message = ref(null)

// When bbox checkbox is toggled, notify parent about drawing state
watch(useBbox, (val) => {
  if (val && bboxMode.value === 'draw') {
    emit('drawing-toggle', true)
  } else {
    emit('drawing-toggle', false)
  }
})

function setBboxMode(mode) {
  bboxMode.value = mode
  emit('drawing-toggle', mode === 'draw' && useBbox.value)
}

// Receive bbox from VideoStream drag
watch(() => props.selectedBbox, (newBbox) => {
  if (newBbox && useBbox.value && bboxMode.value === 'draw') {
    drawnBbox.value = newBbox
    // Also sync to input fields
    bbox.x = newBbox[0]
    bbox.y = newBbox[1]
    bbox.w = newBbox[2]
    bbox.h = newBbox[3]
  }
})

function onSubmit() {
  if (!text.value.trim()) return

  const result = {
    text: text.value.trim()
  }

  if (useBbox.value) {
    if (bboxMode.value === 'draw' && drawnBbox.value) {
      result.bbox = [...drawnBbox.value]
    } else if (bboxMode.value === 'input' && bbox.x !== null && bbox.y !== null && bbox.w !== null && bbox.h !== null) {
      result.bbox = [bbox.x, bbox.y, bbox.w, bbox.h]
    }
  }

  emit('submit', result)

  message.value = {
    type: result.bbox ? 'success' : 'info',
    text: result.bbox
      ? 'Target switch queued!'
      : 'Text updated (no bbox: tracker not re-initialized)'
  }

  setTimeout(() => { message.value = null }, 3000)

  // Reset
  text.value = ''
  drawnBbox.value = null
  emit('drawing-toggle', false)
}

function focus() {
  textInput.value?.focus()
}

defineExpose({
  focus
})
</script>

<style scoped>
.target-input__form {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.target-input__bbox-mode {
  display: flex;
  gap: var(--spacing-xs);
  margin-bottom: var(--spacing-sm);
}

.target-input__mode-btn {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-xs);
  padding: 6px 8px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  cursor: pointer;
  font-size: var(--font-size-sm);
  transition: all 0.15s;
}

.target-input__mode-btn:hover:not(:disabled) {
  border-color: var(--color-text-secondary);
}

.target-input__mode-btn--active {
  border-color: var(--color-accent-primary);
  background: rgba(31, 111, 235, 0.1);
  color: var(--color-accent-primary);
}

.target-input__draw-info {
  padding: var(--spacing-sm);
  background: var(--color-bg-primary);
  border-radius: var(--radius-sm);
  font-size: var(--font-size-sm);
}

.target-input__drawn-bbox {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.target-input__bbox-inputs {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--spacing-sm);
}

.status-message {
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--radius-sm);
  font-size: var(--font-size-sm);
  text-align: center;
}

.status-message.success {
  background: rgba(35, 134, 54, 0.15);
  color: #3fb950;
  border: 1px solid rgba(35, 134, 54, 0.4);
}

.status-message.error {
  background: rgba(218, 54, 51, 0.15);
  color: #f85149;
  border: 1px solid rgba(218, 54, 51, 0.4);
}

.status-message.info {
  background: rgba(31, 111, 235, 0.15);
  color: #58a6ff;
  border: 1px solid rgba(31, 111, 235, 0.4);
}
</style>
