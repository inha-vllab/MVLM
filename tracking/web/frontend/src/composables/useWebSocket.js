/**
 * WebSocket Connection Composable
 *
 * Manages WebSocket connection for real-time tracker status updates.
 */

import { ref } from 'vue'

export function useWebSocket() {
  const ws = ref(null)
  const connected = ref(false)
  const status = ref(null)
  const reconnectAttempts = ref(0)
  const maxReconnectAttempts = 5
  const reconnectDelay = 2000

  // Event handlers
  const onStatusUpdate = ref(null)
  const onConnected = ref(null)
  const onDisconnected = ref(null)
  const onError = ref(null)

  /**
   * Connect to WebSocket
   */
  function connect(url = `ws://${window.location.host}/ws`) {
    if (ws.value && (ws.value.readyState === WebSocket.CONNECTING || ws.value.readyState === WebSocket.OPEN)) {
      return
    }

    try {
      // Use the current host with ws:// or wss:// protocol
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.hostname}:8080/ws`

      ws.value = new WebSocket(wsUrl)

      ws.value.onopen = () => {
        connected.value = true
        reconnectAttempts.value = 0
        console.log('WebSocket connected')
        if (onConnected.value) onConnected.value()
      }

      ws.value.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)

          if (data.type === 'status' || data.type === 'status_update') {
            status.value = data.data
            if (onStatusUpdate.value) onStatusUpdate.value(data.data)
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }

      ws.value.onerror = (error) => {
        console.error('WebSocket error:', error)
        if (onError.value) onError.value(error)
      }

      ws.value.onclose = () => {
        connected.value = false
        console.log('WebSocket disconnected')
        if (onDisconnected.value) onDisconnected.value()

        // Attempt to reconnect
        if (reconnectAttempts.value < maxReconnectAttempts) {
          reconnectAttempts.value++
          console.log(`Reconnecting... attempt ${reconnectAttempts.value}/${maxReconnectAttempts}`)
          setTimeout(connect, reconnectDelay)
        }
      }
    } catch (e) {
      console.error('Failed to create WebSocket:', e)
      if (onError.value) onError.value(e)
    }
  }

  /**
   * Disconnect from WebSocket
   */
  function disconnect() {
    if (ws.value) {
      reconnectAttempts.value = maxReconnectAttempts // Prevent reconnection
      ws.value.close()
      ws.value = null
      connected.value = false
    }
  }

  /**
   * Send message through WebSocket
   */
  function send(data) {
    if (ws.value && ws.value.readyState === WebSocket.OPEN) {
      ws.value.send(JSON.stringify(data))
    }
  }

  return {
    ws,
    connected,
    status,
    connect,
    disconnect,
    send,
    onStatusUpdate,
    onConnected,
    onDisconnected,
    onError
  }
}
