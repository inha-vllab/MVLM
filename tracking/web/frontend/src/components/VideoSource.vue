<template>
  <div class="video-source">
    <div class="panel-header">Video Source</div>

    <div class="video-source__tabs">
      <button
        v-for="source in sources"
        :key="source.id"
        @click="activeSource = source.id"
        :class="['video-source__tab', { 'video-source__tab--active': activeSource === source.id }]"
      >
        <component :is="source.icon" width="16" height="16" />
        {{ source.label }}
      </button>
    </div>

    <!-- Server Path -->
    <div v-if="activeSource === 'path'" class="video-source__content">
      <div class="form-group">
        <label class="label">Server File Path</label>
        <input
          v-model="serverPath"
          type="text"
          class="input"
          placeholder="e.g., /path/to/video.mp4"
        />
      </div>

      <button
        @click="onLoadPath"
        class="btn btn-primary w-full"
        :disabled="!serverPath || loading"
      >
        Load Video
      </button>
    </div>

    <!-- File Upload -->
    <div v-if="activeSource === 'file'" class="video-source__content">
      <div
        class="video-source__dropzone"
        :class="{ 'video-source__dropzone--dragover': isDragover }"
        @dragover.prevent="isDragover = true"
        @dragleave.prevent="isDragover = false"
        @drop.prevent="onDrop"
      >
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17,8 12,3 7,8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
        <p class="text-muted">Drop video file here or click to browse</p>
        <input
          ref="fileInputRef"
          type="file"
          accept="video/*"
          @change="onFileSelect"
          style="display: none;"
        />
        <button @click="fileInputRef?.click()" class="btn btn-sm btn-primary" style="margin-top: 12px;">
          Browse Files
        </button>
      </div>

      <div v-if="selectedFile" class="video-source__file-info">
        <span class="text-muted">Selected:</span>
        <span>{{ selectedFile.name }}</span>
        <span class="text-muted">({{ formatFileSize(selectedFile.size) }})</span>
      </div>

      <button
        @click="onLoadFile"
        class="btn btn-primary w-full"
        :disabled="!selectedFile || loading"
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
        Load Video
      </button>
    </div>

    <!-- URL Input -->
    <div v-else-if="activeSource === 'url'" class="video-source__content">
      <div class="form-group">
        <label class="label">Video URL</label>
        <input
          v-model="videoUrl"
          type="text"
          class="input"
          placeholder="https://example.com/video.mp4"
        />
      </div>

      <button
        @click="onLoadUrl"
        class="btn btn-primary w-full"
        :disabled="!videoUrl || loading"
      >
        Load from URL
      </button>
    </div>

    <!-- Webcam -->
    <div v-else-if="activeSource === 'webcam'" class="video-source__content">
      <div class="form-group">
        <label class="label">Device</label>
        <select v-model="webcamDevice" class="select">
          <option :value="0">Default Camera (/dev/video0)</option>
          <option :value="1">Camera 1 (/dev/video1)</option>
          <option :value="2">Camera 2 (/dev/video2)</option>
        </select>
      </div>

      <button
        @click="onLoadWebcam"
        class="btn btn-primary w-full"
        :disabled="loading"
      >
        Open Webcam
      </button>
    </div>

    <!-- Target Description for Starting -->
    <div v-if="videoLoaded" class="video-source__start">
      <div style="border-top: 1px solid var(--color-border); padding-top: 16px; margin-top: 16px;">
        <label class="label">Target Description</label>
        <input
          v-model="targetText"
          type="text"
          class="input"
          placeholder="e.g., the person in blue shirt"
          style="margin-bottom: 8px;"
        />
        <template v-if="!bboxDisabled">
          <p v-if="selectedBbox" class="text-muted" style="font-size: 12px; margin-bottom: 8px;">
            ROI: [{{ selectedBbox.join(', ') }}]
          </p>
          <p v-else class="text-muted" style="font-size: 12px; margin-bottom: 8px;">
            Drag on the preview to select initial ROI (optional)
          </p>
        </template>
        <p v-else class="text-muted" style="font-size: 12px; margin-bottom: 8px;">
          Template-Free mode: bbox not required
        </p>

        <button
          @click="onStart"
          class="btn btn-success w-full"
          :disabled="!targetText || loading"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <polygon points="5,3 19,12 5,21"/>
          </svg>
          Start Tracking
        </button>
      </div>
    </div>

    <div v-if="message" :class="['status-message', message.type]">
      {{ message.text }}
    </div>
  </div>
</template>

<script setup>
import { ref, h } from 'vue'
import { useTracker } from '../composables/useTracker.js'

const props = defineProps({
  isRunning: {
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

const emit = defineEmits(['loaded', 'start'])

const { loadVideo, uploadVideo, startTracking, loading } = useTracker()

const sources = [
  {
    id: 'path',
    label: 'Path',
    icon: () => h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }, [
      h('path', { d: 'M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z' })
    ])
  },
  {
    id: 'file',
    label: 'Upload',
    icon: () => h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }, [
      h('path', { d: 'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z' }),
      h('polyline', { points: '14,2 14,8 20,8' })
    ])
  },
  {
    id: 'url',
    label: 'URL',
    icon: () => h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }, [
      h('path', { d: 'M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71' }),
      h('path', { d: 'M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71' })
    ])
  },
  {
    id: 'webcam',
    label: 'Webcam',
    icon: () => h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }, [
      h('path', { d: 'M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z' }),
      h('circle', { cx: '12', cy: '13', r: '4' })
    ])
  }
]

const activeSource = ref('path')
const fileInputRef = ref(null)
const isDragover = ref(false)
const selectedFile = ref(null)
const serverPath = ref('')
const videoUrl = ref('')
const webcamDevice = ref(0)
const targetText = ref('')
const videoLoaded = ref(false)
const message = ref(null)

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / 1024 / 1024).toFixed(1) + ' MB'
}

function onDrop(e) {
  isDragover.value = false
  const files = e.dataTransfer.files
  if (files.length > 0 && files[0].type.startsWith('video/')) {
    selectedFile.value = files[0]
  }
}

function onFileSelect(e) {
  if (e.target.files.length > 0) {
    selectedFile.value = e.target.files[0]
  }
}

async function onLoadPath() {
  if (!serverPath.value) return

  try {
    await loadVideo('file', serverPath.value)
    videoLoaded.value = true
    showMessage('success', 'Video loaded successfully')
    emit('loaded')
  } catch (e) {
    showMessage('error', 'Failed to load video: ' + e.message)
  }
}

async function onLoadFile() {
  if (!selectedFile.value) return

  try {
    // Upload first, then load from the returned server path
    const uploadResult = await uploadVideo(selectedFile.value)
    await loadVideo('file', uploadResult.path)
    videoLoaded.value = true
    showMessage('success', 'Video loaded successfully')
    emit('loaded')
  } catch (e) {
    showMessage('error', 'Failed to load video: ' + e.message)
  }
}

async function onLoadUrl() {
  if (!videoUrl.value) return

  try {
    await loadVideo('url', null, videoUrl.value)
    videoLoaded.value = true
    showMessage('success', 'Video loaded successfully')
    emit('loaded')
  } catch (e) {
    showMessage('error', 'Failed to load video: ' + e.message)
  }
}

async function onLoadWebcam() {
  try {
    await loadVideo('webcam', null, null, webcamDevice.value)
    videoLoaded.value = true
    showMessage('success', 'Webcam opened')
    emit('loaded')
  } catch (e) {
    showMessage('error', 'Failed to open webcam: ' + e.message)
  }
}

async function onStart() {
  if (!targetText.value) return

  try {
    const bbox = props.selectedBbox || null
    await startTracking(targetText.value, bbox)
    showMessage('success', 'Tracking started')
    emit('start', { text: targetText.value, bbox })
  } catch (e) {
    showMessage('error', 'Failed to start tracking: ' + e.message)
  }
}

function showMessage(type, text) {
  message.value = { type, text }
  setTimeout(() => { message.value = null }, 3000)
}
</script>

<style scoped>
.video-source {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.video-source__tabs {
  display: flex;
  gap: var(--spacing-xs);
}

.video-source__tab {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-sm);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  cursor: pointer;
  font-size: var(--font-size-sm);
  transition: all 0.15s;
}

.video-source__tab:hover {
  border-color: var(--color-text-secondary);
}

.video-source__tab--active {
  border-color: var(--color-accent-primary);
  background: rgba(31, 111, 235, 0.1);
  color: var(--color-accent-primary);
}

.video-source__content {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.video-source__dropzone {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: var(--spacing-xl);
  border: 2px dashed var(--color-border);
  border-radius: var(--radius-md);
  color: var(--color-text-muted);
  transition: all 0.15s;
}

.video-source__dropzone--dragover {
  border-color: var(--color-accent-primary);
  background: rgba(31, 111, 235, 0.05);
}

.video-source__file-info {
  display: flex;
  gap: var(--spacing-xs);
  font-size: var(--font-size-sm);
  padding: var(--spacing-sm);
  background: var(--color-bg-primary);
  border-radius: var(--radius-sm);
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
</style>
