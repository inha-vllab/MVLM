<template>
  <div class="app">
    <header class="app-header">
      <div class="app-header__title">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <circle cx="12" cy="12" r="3"/>
          <line x1="12" y1="2" x2="12" y2="4"/>
          <line x1="12" y1="20" x2="12" y2="22"/>
          <line x1="4.93" y1="4.93" x2="6.34" y2="6.34"/>
          <line x1="17.66" y1="17.66" x2="19.07" y2="19.07"/>
          <line x1="2" y1="12" x2="4" y2="12"/>
          <line x1="20" y1="12" x2="22" y2="12"/>
          <line x1="6.34" y1="17.66" x2="4.93" y2="19.07"/>
          <line x1="19.07" y1="4.93" x2="17.66" y2="6.34"/>
        </svg>
        <span>MVLM Tracker Web UI</span>
      </div>

      <!-- Step Indicator -->
      <div class="app-header__steps">
        <button
          v-for="(step, idx) in steps"
          :key="step.id"
          @click="activeTab = step.tab"
          :class="[
            'app-header__step',
            { 'app-header__step--active': activeTab === step.tab },
            { 'app-header__step--done': step.done }
          ]"
        >
          <span class="app-header__step-num">
            <svg v-if="step.done" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
            <template v-else>{{ idx + 1 }}</template>
          </span>
          {{ step.label }}
        </button>
      </div>

      <div class="app-header__actions">
        <button
          v-if="status?.is_running"
          @click="activeTab = 'control'"
          :class="['btn btn-sm', activeTab === 'control' ? 'btn-primary' : '']"
        >
          Control
        </button>
        <button
          @click="activeTab = 'settings'"
          :class="['btn btn-sm', activeTab === 'settings' ? 'btn-primary' : '']"
        >
          Settings
        </button>
      </div>
    </header>

    <main class="app-main">
      <div class="app-layout">
        <!-- Video Section -->
        <div class="video-section">
          <VideoStream
            ref="videoStreamRef"
            :stream-url="streamUrl"
            :preview-url="previewUrl"
            :stream-active="status?.is_running"
            :video-width="status?.video_width || 0"
            :video-height="status?.video_height || 0"
            :drawing-enabled="!isTemplateFree && bboxDrawingEnabled"
            :bbox-disabled="isTemplateFree"
            @bbox-selected="onBboxSelected"
          />

          <StatusBar :status="status" />
        </div>

        <!-- Control Panel -->
        <div class="control-section">
          <div v-if="activeTab === 'control'" class="panel">
            <ControlPanel
              :is-running="status?.is_running"
              :is-paused="status?.is_paused"
              :current-target="status?.current_text"
              :current-frame="status?.current_frame"
              :total-frames="status?.total_frames"
              :fps="status?.fps"
              @pause="onPause"
              @quit="onQuit"
            />

            <div style="margin-top: 16px;">
              <TargetInput
                ref="targetInputRef"
                :disabled="!status?.is_running"
                :selected-bbox="selectedBbox"
                :bbox-disabled="isTemplateFree"
                @submit="onSwitchTarget"
                @drawing-toggle="onDrawingToggle"
              />
            </div>
          </div>

          <div v-else-if="activeTab === 'model'" class="panel">
            <ModelSelector
              :current-model="currentModel"
              @loaded="onModelLoaded"
            />
          </div>

          <div v-else-if="activeTab === 'settings'" class="panel">
            <SettingsPanel />
          </div>

          <div v-else-if="activeTab === 'video'" class="panel">
            <VideoSource
              :is-running="status?.is_running"
              :selected-bbox="selectedBbox"
              :bbox-disabled="isTemplateFree"
              @loaded="onVideoLoaded"
              @start="onStartTracking"
            />
          </div>
        </div>
      </div>
    </main>

    <!-- Connection Status Toast -->
    <div v-if="showConnectionStatus" :class="['connection-toast', connectionStatusClass]">
      <span>{{ connectionMessage }}</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useTracker } from './composables/useTracker.js'
import { useWebSocket } from './composables/useWebSocket.js'
import VideoStream from './components/VideoStream.vue'
import StatusBar from './components/StatusBar.vue'
import ControlPanel from './components/ControlPanel.vue'
import TargetInput from './components/TargetInput.vue'
import ModelSelector from './components/ModelSelector.vue'
import SettingsPanel from './components/SettingsPanel.vue'
import VideoSource from './components/VideoSource.vue'

const { getStatus, togglePause, quit, switchTarget, startTracking } = useTracker()
const { connect, disconnect, status: wsStatus, connected, onStatusUpdate, onConnected, onDisconnected } = useWebSocket()

const videoStreamRef = ref(null)
const targetInputRef = ref(null)

const activeTab = ref('model')

const status = ref(null)
const currentModel = ref(null)
const streamUrl = ref('/stream')
const previewUrl = ref(null)
const selectedBbox = ref(null)
const bboxDrawingEnabled = ref(false)

// TF mode: config name contains "_TF" → bbox is not used
const isTemplateFree = computed(() => {
  const cfg = status.value?.config || currentModel.value?.config || ''
  return cfg.toUpperCase().includes('_TF')
})

// Step indicators (tab maps step click to the correct panel)
const steps = computed(() => [
  { id: 'step-model', tab: 'model', label: 'Model', done: !!status.value?.model_loaded },
  { id: 'step-video', tab: 'video', label: 'Video', done: !!status.value?.video_loaded },
  { id: 'step-track', tab: 'video', label: 'Track', done: !!status.value?.is_running },
])

const showConnectionStatus = ref(false)
const connectionMessage = ref('')
const connectionStatusClass = ref('')

onMounted(() => {
  // Connect WebSocket
  onConnected.value = () => {
    showConnectionStatus.value = true
    connectionMessage.value = 'Connected to tracker'
    connectionStatusClass.value = 'connection-toast--success'
    setTimeout(() => { showConnectionStatus.value = false }, 2000)
  }

  onDisconnected.value = () => {
    showConnectionStatus.value = true
    connectionMessage.value = 'Disconnected from tracker'
    connectionStatusClass.value = 'connection-toast--error'
  }

  onStatusUpdate.value = (data) => {
    status.value = data
    if (videoStreamRef.value) {
      videoStreamRef.value.updateOverlay(data)
    }
  }

  connect()

  // Initial status fetch
  getStatus().then(data => {
    status.value = data
    // Auto-navigate to the first incomplete step
    if (!data?.model_loaded) {
      activeTab.value = 'model'
    } else if (!data?.video_loaded) {
      activeTab.value = 'video'
    } else if (data?.is_running) {
      activeTab.value = 'control'
    }
  })
})

onUnmounted(() => {
  disconnect()
})

async function onPause() {
  try {
    await togglePause()
    status.value = await getStatus()
  } catch (e) {
    console.error('Pause error:', e)
  }
}

async function onQuit() {
  try {
    await quit()
    status.value = await getStatus()
    previewUrl.value = null
    selectedBbox.value = null
  } catch (e) {
    console.error('Quit error:', e)
  }
}

async function onSwitchTarget(data) {
  try {
    await switchTarget(data.text, data.bbox)
    status.value = await getStatus()
  } catch (e) {
    console.error('Switch target error:', e)
  }
}

function onModelLoaded(model) {
  currentModel.value = model
  // Auto-advance to video tab
  activeTab.value = 'video'
}

function onVideoLoaded() {
  // Show first frame preview
  previewUrl.value = '/api/video/first_frame?t=' + Date.now()
  selectedBbox.value = null
}

function onBboxSelected(bbox) {
  selectedBbox.value = bbox
}

function onDrawingToggle(enabled) {
  bboxDrawingEnabled.value = enabled
  if (!enabled) {
    // Clear any drawn bbox on video when drawing is disabled
    selectedBbox.value = null
  }
}

async function onStartTracking(data) {
  try {
    // startTracking is already called inside VideoSource.vue
    // Just update state and switch to control tab
    status.value = await getStatus()
    activeTab.value = 'control'
    previewUrl.value = null
  } catch (e) {
    console.error('Start tracking error:', e)
  }
}
</script>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--color-bg-primary);
}

.app-header {
  display: flex;
  align-items: center;
  padding: 0 var(--spacing-lg);
  height: 56px;
  background: var(--color-bg-secondary);
  border-bottom: 1px solid var(--color-border);
  gap: var(--spacing-xl);
}

.app-header__title {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-weight: 600;
  font-size: var(--font-size-lg);
  white-space: nowrap;
}

.app-header__steps {
  display: flex;
  gap: var(--spacing-xs);
  flex: 1;
}

.app-header__step {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-sm) var(--spacing-md);
  border: none;
  background: transparent;
  color: var(--color-text-secondary);
  cursor: pointer;
  border-radius: var(--radius-sm);
  font-size: var(--font-size-sm);
  transition: color 0.15s, background 0.15s;
}

.app-header__step:hover {
  color: var(--color-text-primary);
  background: var(--color-bg-tertiary);
}

.app-header__step--active {
  color: var(--color-accent-primary);
  background: rgba(31, 111, 235, 0.1);
}

.app-header__step--done .app-header__step-num {
  background: var(--color-success);
  color: white;
}

.app-header__step-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: var(--color-bg-tertiary);
  font-size: 11px;
  font-weight: 600;
  flex-shrink: 0;
}

.app-header__actions {
  display: flex;
  gap: var(--spacing-sm);
}

.app-main {
  flex: 1;
  overflow: hidden;
}

.app-layout {
  display: grid;
  grid-template-columns: 1fr 360px;
  gap: var(--spacing-lg);
  height: 100%;
  padding: var(--spacing-lg);
}

.video-section {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  min-height: 0;
}

.control-section {
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow-y: auto;
}

.connection-toast {
  position: fixed;
  bottom: var(--spacing-lg);
  left: 50%;
  transform: translateX(-50%);
  padding: var(--spacing-sm) var(--spacing-lg);
  border-radius: var(--radius-md);
  font-size: var(--font-size-sm);
  z-index: 1000;
  animation: slideUp 0.3s ease-out;
}

.connection-toast--success {
  background: var(--color-success);
  color: white;
}

.connection-toast--error {
  background: var(--color-danger);
  color: white;
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateX(-50%) translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }
}

@media (max-width: 1024px) {
  .app-layout {
    grid-template-columns: 1fr;
    grid-template-rows: 1fr auto;
  }
}
</style>
