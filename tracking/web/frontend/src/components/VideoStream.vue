<template>
  <div class="video-stream" :class="{ 'video-stream--loading': loading }" ref="containerRef">
    <!-- Live MJPEG stream (when tracking is running) -->
    <img
      v-if="streamActive"
      ref="imageRef"
      :src="streamUrl"
      alt="Video Stream"
      @load="onLoad"
      @error="onError"
      crossorigin="anonymous"
    />

    <!-- First frame preview (when video is loaded but not tracking) -->
    <img
      v-else-if="previewUrl"
      ref="previewImageRef"
      :src="previewUrl"
      alt="First Frame Preview"
      class="video-stream__preview-img"
      @load="onPreviewLoad"
    />

    <!-- Placeholder (no video loaded) -->
    <div v-else class="video-stream__placeholder">
      <div class="video-stream__placeholder-content">
        <svg v-if="!error" class="video-stream__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <circle cx="12" cy="12" r="10" stroke-width="2"/>
          <circle cx="12" cy="12" r="4" fill="currentColor"/>
        </svg>
        <svg v-else class="video-stream__icon video-stream__icon--error" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <circle cx="12" cy="12" r="10" stroke-width="2"/>
          <line x1="15" y1="9" x2="9" y2="15" stroke-width="2"/>
          <line x1="9" y1="9" x2="15" y2="15" stroke-width="2"/>
        </svg>
        <p class="text-muted">{{ error ? 'Stream Error' : 'Load a video to preview' }}</p>
        <button v-if="error" @click="$emit('retry')" class="btn btn-sm btn-primary" style="margin-top: 12px;">
          Retry
        </button>
      </div>
    </div>

    <!-- Bbox drawing canvas overlay (available on both preview and live stream) -->
    <canvas
      v-if="canvasVisible"
      ref="bboxCanvasRef"
      class="video-stream__bbox-canvas"
      :class="{ 'video-stream__bbox-canvas--active': drawingEnabled }"
      @mousedown="onCanvasMouseDown"
      @mousemove="onCanvasMouseMove"
      @mouseup="onCanvasMouseUp"
      @mouseleave="onCanvasMouseUp"
    />

    <!-- Hints and controls for bbox drawing -->
    <div v-if="canvasVisible && drawingEnabled && !drawnBbox" class="video-stream__hint">
      Drag to select target ROI
    </div>
    <button
      v-if="canvasVisible && drawnBbox"
      @click="clearBbox"
      class="video-stream__clear-btn"
    >
      Clear
    </button>

    <!-- Overlay stats (during tracking) -->
    <div v-if="overlayData && streamActive" class="video-stream__overlay">
      <div class="video-stream__overlay-item">
        <span class="text-muted">Frame:</span>
        <span>{{ overlayData.current_frame || 0 }} / {{ overlayData.total_frames || '?' }}</span>
      </div>
      <div class="video-stream__overlay-item">
        <span class="text-muted">FPS:</span>
        <span :class="fpsClass">{{ overlayData.fps?.toFixed(1) || '0.0' }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'

const props = defineProps({
  streamUrl: {
    type: String,
    default: '/stream'
  },
  streamActive: {
    type: Boolean,
    default: false
  },
  previewUrl: {
    type: String,
    default: null
  },
  videoWidth: {
    type: Number,
    default: 0
  },
  videoHeight: {
    type: Number,
    default: 0
  },
  drawingEnabled: {
    type: Boolean,
    default: false
  },
  bboxDisabled: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['load', 'error', 'retry', 'bbox-selected'])

const containerRef = ref(null)
const imageRef = ref(null)
const previewImageRef = ref(null)
const bboxCanvasRef = ref(null)
const loading = ref(false)
const error = ref(null)
const overlayData = ref(null)

// Bbox drawing state
const isDrawing = ref(false)
const drawStart = ref(null)
const drawnBbox = ref(null)

// Canvas is visible when there's something to draw on (preview or stream)
// and drawing is enabled, OR preview mode (always allow drawing on preview)
const canvasVisible = computed(() => {
  if (props.bboxDisabled) return false  // TF mode: no bbox needed
  if (props.previewUrl && !props.streamActive) return true  // preview mode
  if (props.streamActive && props.drawingEnabled) return true  // live + draw mode
  return false
})

const fpsClass = computed(() => {
  const fps = overlayData.value?.fps || 0
  if (fps < 10) return 'badge badge-danger'
  if (fps < 20) return 'badge badge-warning'
  return 'badge badge-success'
})

// Resize canvas when drawing becomes enabled or stream state changes
watch([() => props.drawingEnabled, () => props.streamActive, () => props.previewUrl], () => {
  nextTick(() => resizeCanvas())
}, { flush: 'post' })

function onLoad() {
  loading.value = false
  error.value = null
  emit('load')
}

function onError(e) {
  loading.value = false
  error.value = e
  emit('error', e)
}

function onPreviewLoad() {
  nextTick(() => resizeCanvas())
}

function resizeCanvas() {
  const canvas = bboxCanvasRef.value
  const container = containerRef.value
  if (!canvas || !container) return

  const rect = container.getBoundingClientRect()
  canvas.width = rect.width
  canvas.height = rect.height

  if (drawnBbox.value) {
    drawBboxOnCanvas(drawnBbox.value)
  }
}

function getCanvasCoords(e) {
  const canvas = bboxCanvasRef.value
  if (!canvas) return null
  const rect = canvas.getBoundingClientRect()
  return {
    x: e.clientX - rect.left,
    y: e.clientY - rect.top
  }
}

function canvasToVideo(canvasX, canvasY) {
  const canvas = bboxCanvasRef.value
  if (!canvas) return { x: canvasX, y: canvasY }

  const vw = props.videoWidth
  const vh = props.videoHeight
  if (!vw || !vh) return { x: canvasX, y: canvasY }

  // Compute the displayed image area (object-fit: contain)
  const cw = canvas.width
  const ch = canvas.height
  const scale = Math.min(cw / vw, ch / vh)
  const imgW = vw * scale
  const imgH = vh * scale
  const offsetX = (cw - imgW) / 2
  const offsetY = (ch - imgH) / 2

  return {
    x: Math.round((canvasX - offsetX) / scale),
    y: Math.round((canvasY - offsetY) / scale)
  }
}

function onCanvasMouseDown(e) {
  // Only allow drawing when enabled (preview always enabled, stream needs prop)
  if (props.streamActive && !props.drawingEnabled) return

  const coords = getCanvasCoords(e)
  if (!coords) return
  isDrawing.value = true
  drawStart.value = coords
  drawnBbox.value = null
  emit('bbox-selected', null)
}

function onCanvasMouseMove(e) {
  if (!isDrawing.value || !drawStart.value) return
  const coords = getCanvasCoords(e)
  if (!coords) return

  const canvas = bboxCanvasRef.value
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  const x = Math.min(drawStart.value.x, coords.x)
  const y = Math.min(drawStart.value.y, coords.y)
  const w = Math.abs(coords.x - drawStart.value.x)
  const h = Math.abs(coords.y - drawStart.value.y)

  ctx.strokeStyle = '#00ff00'
  ctx.lineWidth = 2
  ctx.setLineDash([6, 3])
  ctx.strokeRect(x, y, w, h)
  ctx.fillStyle = 'rgba(0, 255, 0, 0.1)'
  ctx.fillRect(x, y, w, h)
}

function onCanvasMouseUp(e) {
  if (!isDrawing.value || !drawStart.value) return
  isDrawing.value = false

  const coords = getCanvasCoords(e)
  if (!coords) return

  const canvasX = Math.min(drawStart.value.x, coords.x)
  const canvasY = Math.min(drawStart.value.y, coords.y)
  const canvasW = Math.abs(coords.x - drawStart.value.x)
  const canvasH = Math.abs(coords.y - drawStart.value.y)

  // Ignore tiny drags
  if (canvasW < 5 || canvasH < 5) {
    const canvas = bboxCanvasRef.value
    if (canvas) {
      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)
    }
    drawStart.value = null
    return
  }

  const topLeft = canvasToVideo(canvasX, canvasY)
  const bottomRight = canvasToVideo(canvasX + canvasW, canvasY + canvasH)

  const videoBbox = [
    Math.max(0, topLeft.x),
    Math.max(0, topLeft.y),
    bottomRight.x - topLeft.x,
    bottomRight.y - topLeft.y
  ]

  drawnBbox.value = { canvasX, canvasY, canvasW, canvasH }
  drawBboxOnCanvas(drawnBbox.value)

  emit('bbox-selected', videoBbox)
  drawStart.value = null
}

function drawBboxOnCanvas(bbox) {
  const canvas = bboxCanvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  ctx.strokeStyle = '#00ff00'
  ctx.lineWidth = 2
  ctx.setLineDash([])
  ctx.strokeRect(bbox.canvasX, bbox.canvasY, bbox.canvasW, bbox.canvasH)
  ctx.fillStyle = 'rgba(0, 255, 0, 0.1)'
  ctx.fillRect(bbox.canvasX, bbox.canvasY, bbox.canvasW, bbox.canvasH)
}

function clearBbox() {
  drawnBbox.value = null
  const canvas = bboxCanvasRef.value
  if (canvas) {
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
  }
  emit('bbox-selected', null)
}

function updateOverlay(data) {
  overlayData.value = data
}

// Refresh stream periodically
let refreshInterval
let resizeObserver

onMounted(() => {
  refreshInterval = setInterval(() => {
    if (imageRef.value && props.streamActive) {
      imageRef.value.src = props.streamUrl + '?t=' + Date.now()
    }
  }, 100)

  resizeObserver = new ResizeObserver(() => resizeCanvas())
  if (containerRef.value) {
    resizeObserver.observe(containerRef.value)
  }
})

onUnmounted(() => {
  if (refreshInterval) clearInterval(refreshInterval)
  if (resizeObserver) resizeObserver.disconnect()
})

defineExpose({
  updateOverlay
})
</script>

<style scoped>
.video-stream {
  position: relative;
  background: #000;
  border-radius: var(--radius-md);
  overflow: hidden;
  aspect-ratio: 16 / 9;
}

.video-stream img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.video-stream__preview-img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.video-stream__bbox-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  cursor: crosshair;
  z-index: 2;
}

.video-stream__bbox-canvas--active {
  cursor: crosshair;
}

.video-stream__hint {
  position: absolute;
  top: 8px;
  left: 50%;
  transform: translateX(-50%);
  padding: 4px 12px;
  background: rgba(0, 0, 0, 0.6);
  color: rgba(255, 255, 255, 0.8);
  border-radius: var(--radius-sm);
  font-size: var(--font-size-sm);
  pointer-events: none;
  z-index: 3;
}

.video-stream__clear-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  padding: 4px 10px;
  background: rgba(0, 0, 0, 0.6);
  color: white;
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: var(--radius-sm);
  font-size: var(--font-size-sm);
  cursor: pointer;
  transition: background 0.15s;
  z-index: 3;
}

.video-stream__clear-btn:hover {
  background: rgba(0, 0, 0, 0.8);
}

.video-stream__placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-bg-primary);
}

.video-stream__placeholder-content {
  text-align: center;
}

.video-stream__icon {
  width: 48px;
  height: 48px;
  margin: 0 auto 16px;
  color: var(--color-text-muted);
}

.video-stream__icon--error {
  color: var(--color-danger-hover);
}

.video-stream__overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 8px 12px;
  background: linear-gradient(transparent, rgba(0,0,0,0.7));
  display: flex;
  gap: 16px;
  font-size: var(--font-size-sm);
  z-index: 1;
}

.video-stream__overlay-item {
  display: flex;
  gap: 4px;
}

.video-stream--loading img {
  opacity: 0;
}
</style>
