import type { SettingsT } from './editor-session.d';

export type PresetT = { id: string; name: string; params: SettingsT };
export type SavedPresetsT = { presets: PresetT[]; skippedInvalid: boolean };
