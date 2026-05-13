/**
 * Settings Persistence Composable
 *
 * Manages local storage persistence for Web UI settings.
 */

import { ref } from 'vue'

const SETTINGS_KEY = 'mvlm_webui_settings'

export function useSettings() {
  const settings = ref(null)

  /**
   * Load settings from local storage
   */
  function loadSettings() {
    try {
      const saved = localStorage.getItem(SETTINGS_KEY)
      if (saved) {
        settings.value = JSON.parse(saved)
      } else {
        settings.value = getDefaultSettings()
      }
    } catch (e) {
      console.error('Failed to load settings:', e)
      settings.value = getDefaultSettings()
    }
    return settings.value
  }

  /**
   * Save settings to local storage
   */
  function saveSettings(newSettings) {
    try {
      settings.value = { ...settings.value, ...newSettings }
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings.value))
    } catch (e) {
      console.error('Failed to save settings:', e)
    }
  }

  /**
   * Get a specific setting value
   */
  function getSetting(key, defaultValue = null) {
    if (!settings.value) loadSettings()
    return settings.value?.[key] ?? defaultValue
  }

  /**
   * Set a specific setting value
   */
  function setSetting(key, value) {
    if (!settings.value) loadSettings()
    settings.value[key] = value
    saveSettings(settings.value)
  }

  /**
   * Reset all settings to defaults
   */
  function resetSettings() {
    settings.value = getDefaultSettings()
    saveSettings(settings.value)
  }

  /**
   * Export settings as JSON
   */
  function exportSettings() {
    return JSON.stringify(settings.value || loadSettings(), null, 2)
  }

  /**
   * Import settings from JSON
   */
  function importSettings(jsonString) {
    try {
      const imported = JSON.parse(jsonString)
      settings.value = { ...getDefaultSettings(), ...imported }
      saveSettings(settings.value)
      return true
    } catch (e) {
      console.error('Failed to import settings:', e)
      return false
    }
  }

  /**
   * Get default settings
   */
  function getDefaultSettings() {
    return {
      // Model settings
      lastConfig: 'mvlm_b224_rgbn_xz_clip_full_cmloss_otu_3D-T-V-L_4gpu_b48_0.0003',
      lastCheckpoint: 'models/260126_sutrack_b224_rgbn_xz_clip_full_cmloss_otu_3D-T-V-L_4gpu_b48_0.0003_mean_ep0060.pth.tar',

      // Video settings
      defaultDataset: 'TNL2K',
      frameSkip: 0,
      useHalfPrecision: false,

      // UI settings
      theme: 'dark',
      sidebarCollapsed: false,
      showConsole: false,

      // Tracker parameters
      searchAreaFactor: 2.0,
      updateInterval: 10,
      updateThreshold: 0.5,

      // Visualization
      showScore: true,
      showBbox: true,
      showText: true,
      bboxColor: 'red',
      bboxThickness: 2,
    }
  }

  return {
    settings,
    loadSettings,
    saveSettings,
    getSetting,
    setSetting,
    resetSettings,
    exportSettings,
    importSettings
  }
}
