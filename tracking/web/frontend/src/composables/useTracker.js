/**
 * Tracker API Communication Composable
 *
 * Provides methods for communicating with the MVLM tracker backend API.
 */

import { ref } from 'vue'

const API_BASE = '/api'

export function useTracker() {
  const status = ref(null)
  const error = ref(null)
  const loading = ref(false)

  /**
   * Fetch current tracker status
   */
  async function getStatus() {
    try {
      const response = await fetch(`${API_BASE}/status`)
      if (!response.ok) throw new Error('Failed to fetch status')
      return await response.json()
    } catch (e) {
      error.value = e.message
      throw e
    }
  }

  /**
   * Pause or resume tracking
   */
  async function togglePause() {
    loading.value = true
    try {
      const response = await fetch(`${API_BASE}/control/pause`, { method: 'POST' })
      if (!response.ok) throw new Error('Failed to toggle pause')
      const result = await response.json()
      status.value = await getStatus()
      return result
    } catch (e) {
      error.value = e.message
      throw e
    } finally {
      loading.value = false
    }
  }

  /**
   * Stop tracker
   */
  async function quit() {
    loading.value = true
    try {
      const response = await fetch(`${API_BASE}/control/quit`, { method: 'POST' })
      if (!response.ok) throw new Error('Failed to quit')
      return await response.json()
    } catch (e) {
      error.value = e.message
      throw e
    } finally {
      loading.value = false
    }
  }

  /**
   * Switch to a new target
   */
  async function switchTarget(text, bbox = null) {
    loading.value = true
    try {
      const response = await fetch(`${API_BASE}/target/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, bbox })
      })
      if (!response.ok) throw new Error('Failed to switch target')
      return await response.json()
    } catch (e) {
      error.value = e.message
      throw e
    } finally {
      loading.value = false
    }
  }

  /**
   * List available configs
   */
  async function listConfigs() {
    try {
      const response = await fetch(`${API_BASE}/configs`)
      if (!response.ok) throw new Error('Failed to list configs')
      const data = await response.json()
      return data.configs || []
    } catch (e) {
      error.value = e.message
      throw e
    }
  }

  /**
   * List available models
   */
  async function listModels() {
    try {
      const response = await fetch(`${API_BASE}/models`)
      if (!response.ok) throw new Error('Failed to list models')
      const data = await response.json()
      return data.models || []
    } catch (e) {
      error.value = e.message
      throw e
    }
  }

  /**
   * Load a model
   */
  async function loadModel(config, checkpoint, dataset = 'TNL2K') {
    loading.value = true
    try {
      const response = await fetch(`${API_BASE}/model/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config, checkpoint, dataset })
      })
      if (!response.ok) throw new Error('Failed to load model')
      return await response.json()
    } catch (e) {
      error.value = e.message
      throw e
    } finally {
      loading.value = false
    }
  }

  /**
   * Upload video file
   */
  async function uploadVideo(file) {
    loading.value = true
    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_BASE}/video/upload`, {
        method: 'POST',
        body: formData
      })
      if (!response.ok) throw new Error('Failed to upload video')
      return await response.json()
    } catch (e) {
      error.value = e.message
      throw e
    } finally {
      loading.value = false
    }
  }

  /**
   * Load video from path or URL
   */
  async function loadVideo(sourceType, path = null, url = null, deviceId = 0) {
    loading.value = true
    try {
      const response = await fetch(`${API_BASE}/video/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_type: sourceType, path, url, device_id: deviceId })
      })
      if (!response.ok) throw new Error('Failed to load video')
      return await response.json()
    } catch (e) {
      error.value = e.message
      throw e
    } finally {
      loading.value = false
    }
  }

  /**
   * Start tracking
   */
  async function startTracking(text, bbox = null) {
    loading.value = true
    try {
      const response = await fetch(`${API_BASE}/video/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, bbox })
      })
      if (!response.ok) throw new Error('Failed to start tracking')
      return await response.json()
    } catch (e) {
      error.value = e.message
      throw e
    } finally {
      loading.value = false
    }
  }

  /**
   * Seek to specific frame
   */
  async function seekFrame(frameNumber) {
    loading.value = true
    try {
      const response = await fetch(`${API_BASE}/video/seek/${frameNumber}`, {
        method: 'POST'
      })
      if (!response.ok) throw new Error('Failed to seek frame')
      return await response.json()
    } catch (e) {
      error.value = e.message
      throw e
    } finally {
      loading.value = false
    }
  }

  /**
   * Get available parameters
   */
  async function getParameters() {
    try {
      const response = await fetch(`${API_BASE}/parameters`)
      if (!response.ok) throw new Error('Failed to get parameters')
      return await response.json()
    } catch (e) {
      error.value = e.message
      throw e
    }
  }

  /**
   * Update tracker parameters
   */
  async function updateParameters(parameters) {
    loading.value = true
    try {
      const response = await fetch(`${API_BASE}/parameters`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ parameters })
      })
      if (!response.ok) throw new Error('Failed to update parameters')
      return await response.json()
    } catch (e) {
      error.value = e.message
      throw e
    } finally {
      loading.value = false
    }
  }

  return {
    status,
    error,
    loading,
    getStatus,
    togglePause,
    quit,
    switchTarget,
    listConfigs,
    listModels,
    loadModel,
    uploadVideo,
    loadVideo,
    startTracking,
    seekFrame,
    getParameters,
    updateParameters
  }
}
