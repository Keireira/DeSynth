import type { SettingsT } from '../../../../entities/editor-session';

export type Props = { settings: SettingsT; disabled: boolean; onChange: (patch: Partial<SettingsT>) => void };
