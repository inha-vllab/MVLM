<template>
  <div class="settings-panel">
    <div class="settings-panel__header">
      <div class="panel-header">Tracker Parameters</div>
      <button @click="onReset" class="btn btn-sm">Reset</button>
    </div>

    <div class="settings-panel__groups">
      <div v-for="(group, groupName) in parameters" :key="groupName" class="settings-group">
        <div class="settings-group__name">{{ groupName }}</div>

        <div v-for="(param, paramName) in group" :key="paramName" class="settings-item">
          <label class="settings-item__label">
            {{ formatLabel(paramName) }}
          </label>

          <!-- Range slider for numeric values -->
          <div v-if="param.type === 'float' || param.type === 'int'" class="settings-item__control">
            <input
              v-if="param.type === 'float'"
              v-model.number="values[paramName]"
              type="range"
              class="range"
              :min="param.min"
              :max="param.max"
              :step="param.step || (param.type === 'float' ? 0.1 : 1)"
              @input="onParamChange(paramName, $event.target.value)"
            />
            <input
              v-else
              v-model.number="values[paramName]"
              type="range"
              class="range"
              :min="param.min"
              :max="param.max"
              @input="onParamChange(paramName, $event.target.value)"
            />
            <span class="settings-item__value">{{ values[paramName] }}</span>
          </div>

          <!-- Select for choices -->
          <div v-else-if="param.choices" class="settings-item__control">
            <select
              v-model="values[paramName]"
              class="select"
              @change="onParamChange(paramName, $event.target.value)"
            >
              <option v-for="choice in param.choices" :key="choice" :value="choice">
                {{ choice }}
              </option>
            </select>
          </div>

          <!-- Checkbox for boolean -->
          <div v-else-if="param.type === 'bool'" class="settings-item__control">
            <label class="checkbox">
              <input
                v-model="values[paramName]"
                type="checkbox"
                @change="onParamChange(paramName, $event.target.checked)"
              />
              <span>{{ values[paramName] ? 'Enabled' : 'Disabled' }}</span>
            </label>
          </div>

          <p v-if="param.description" class="settings-item__description">
            {{ param.description }}
          </p>
        </div>
      </div>
    </div>

    <button
      @click="onApply"
      class="btn btn-primary w-full"
      :disabled="!hasChanges"
    >
      Apply Changes
    </button>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { useTracker } from '../composables/useTracker.js'

const { getParameters, updateParameters, loading } = useTracker()

const parameters = ref({})
const values = reactive({})
const originalValues = reactive({})

const hasChanges = computed(() => {
  return Object.keys(values).some(key => values[key] !== originalValues[key])
})

async function loadParameters() {
  try {
    const params = await getParameters()
    parameters.value = params

    // Set default values
    Object.entries(params).forEach(([groupName, group]) => {
      Object.entries(group).forEach(([paramName, param]) => {
        values[paramName] = param.default
        originalValues[paramName] = param.default
      })
    })
  } catch (e) {
    console.error('Failed to load parameters:', e)
  }
}

function formatLabel(name) {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase())
}

function onParamChange(name, value) {
  values[name] = value
}

async function onApply() {
  try {
    await updateParameters(values)
    Object.assign(originalValues, values)
  } catch (e) {
    console.error('Failed to update parameters:', e)
  }
}

function onReset() {
  Object.entries(parameters.value).forEach(([groupName, group]) => {
    Object.entries(group).forEach(([paramName, param]) => {
      values[paramName] = param.default
    })
  })
  onApply()
}

onMounted(() => {
  loadParameters()
})
</script>

<style scoped>
.settings-panel {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  max-height: 500px;
}

.settings-panel__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.settings-panel__groups {
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
}

.settings-group {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.settings-group__name {
  font-size: var(--font-size-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--color-accent-primary);
  padding-bottom: var(--spacing-xs);
  border-bottom: 1px solid var(--color-border);
}

.settings-item {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.settings-item__label {
  font-size: var(--font-size-sm);
  font-weight: 500;
}

.settings-item__control {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.settings-item__value {
  min-width: 50px;
  text-align: right;
  font-family: monospace;
  font-size: var(--font-size-sm);
}

.settings-item__description {
  font-size: var(--font-size-xs);
  color: var(--color-text-muted);
  margin: 0;
}
</style>
