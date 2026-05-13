<template>
  <div class="model-selector">
    <div class="panel-header">Model Management</div>

    <div class="model-selector__current">
      <span class="text-muted">Current:</span>
      <span v-if="currentModel" class="model-selector__current-name">
        {{ currentModel.config }}
      </span>
      <span v-else class="text-muted">No model loaded</span>
    </div>

    <div class="form-group">
      <label class="label">Config</label>
      <select
        v-model="selectedConfig"
        class="select"
        :disabled="loading || configs.length === 0"
      >
        <option value="">-- Select config --</option>
        <option v-for="c in configs" :key="c" :value="c">{{ c }}</option>
      </select>
    </div>

    <div class="form-group">
      <label class="label">Checkpoint</label>
      <select
        v-model="selectedModelId"
        class="select"
        :disabled="loading || models.length === 0"
      >
        <option value="">-- Select checkpoint --</option>
        <option v-for="model in models" :key="model.path" :value="model.path">
          {{ model.name }}
          <template v-if="model.epoch"> (Epoch {{ model.epoch }})</template>
        </option>
      </select>
    </div>

    <div v-if="selectedModel" class="model-selector__info">
      <div v-if="selectedModel.epoch" class="model-selector__info-item">
        <span class="text-muted">Epoch:</span>
        <span>{{ selectedModel.epoch }}</span>
      </div>
      <div class="model-selector__info-item">
        <span class="text-muted">Size:</span>
        <span>{{ (selectedModel.size_mb || 0).toFixed(1) }} MB</span>
      </div>
    </div>

    <div class="flex gap-sm">
      <button
        @click="onLoad"
        class="btn btn-primary flex-1"
        :disabled="loading || !selectedModel || !selectedConfig"
      >
        <svg v-if="loading" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <circle cx="12" cy="12" r="10" stroke-width="2" stroke-dasharray="32" stroke-dashoffset="32">
            <animate attributeName="stroke-dashoffset" dur="1s" repeatCount="indefinite" to="0"/>
          </circle>
        </svg>
        <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7,10 12,15 17,10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Load
      </button>

      <button
        @click="onRefresh"
        class="btn"
        :disabled="loading"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="23 4 23 10 17 10"/>
          <polyline points="1 20 1 14 7 14"/>
          <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
        </svg>
      </button>
    </div>

    <div v-if="message" :class="['status-message', message.type]">
      {{ message.text }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { useTracker } from '../composables/useTracker.js'

const props = defineProps({
  currentModel: {
    type: Object,
    default: null
  }
})

const emit = defineEmits(['loaded'])

const { listModels, listConfigs, loadModel, loading } = useTracker()

const models = ref([])
const configs = ref([])
const selectedModelId = ref('')
const selectedConfig = ref('')
const message = ref(null)

const selectedModel = computed(() => {
  if (!selectedModelId.value) return null
  return models.value.find(m => m.path === selectedModelId.value)
})

// Auto-fill config when model with known config is selected
watch(selectedModel, (model) => {
  if (model && model.config && !selectedConfig.value) {
    selectedConfig.value = model.config
  }
})

async function loadModels() {
  try {
    models.value = await listModels()
  } catch (e) {
    showMessage('error', 'Failed to load models: ' + e.message)
  }
}

async function loadConfigList() {
  try {
    configs.value = await listConfigs()
  } catch (e) {
    showMessage('error', 'Failed to load configs: ' + e.message)
  }
}

async function onLoad() {
  if (!selectedModel.value || !selectedConfig.value) return

  try {
    await loadModel(
      selectedConfig.value,
      selectedModel.value.path,
      'TNL2K'
    )
    showMessage('success', 'Model loaded successfully')
    emit('loaded', { ...selectedModel.value, config: selectedConfig.value })
  } catch (e) {
    showMessage('error', 'Failed to load model: ' + e.message)
  }
}

function onRefresh() {
  loadModels()
  loadConfigList()
  showMessage('info', 'Model list refreshed')
}

function showMessage(type, text) {
  message.value = { type, text }
  setTimeout(() => { message.value = null }, 3000)
}

onMounted(() => {
  loadModels()
  loadConfigList()
})
</script>

<style scoped>
.model-selector {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.model-selector__current {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-sm) 0;
  border-bottom: 1px solid var(--color-border);
  font-size: var(--font-size-sm);
}

.model-selector__current-name {
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 200px;
}

.model-selector__info {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  padding: var(--spacing-sm);
  background: var(--color-bg-primary);
  border-radius: var(--radius-sm);
  font-size: var(--font-size-sm);
}

.model-selector__info-item {
  display: flex;
  justify-content: space-between;
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
