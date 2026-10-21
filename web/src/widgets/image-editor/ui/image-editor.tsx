import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import type { SettingsT } from '../../../entities/editor-session';
import { Button, Icon } from '../../../shared/ui';
import HelpDialog from '../../../features/show-help';
import Settings from '../../../features/edit-settings';
import { configurationProblem } from '../../../entities/editor-session';
import Viewer from '../../../features/compare-images';
import type { ViewerHandleT } from '../../../features/compare-images';
import { GenerationPanel, RestorationPanel } from './index';
import { errorMessage } from '../../../shared/api';
import { useStudio } from '../model/use-studio';
import { Actions, Editor, Feedback, ProcessPanel, SettingsScroll, Shell, Sidebar, Toast } from './image-editor.styles';

const ImageEditor = () => {
	const { state, studio } = useStudio();
	const [settingsValue, setSettingsValue] = useState<SettingsT | null>(null);
	const [helpOpen, setHelpOpen] = useState(false);
	const [exporting, setExporting] = useState(false);
	const fileInput = useRef<HTMLInputElement>(null);
	const viewer = useRef<ViewerHandleT>(null);
	useEffect(() => {
		if (!state.notice) return;
		const timer = setTimeout(studio.dismissNotice, state.notice.error ? 6500 : 3500);
		return () => {
			clearTimeout(timer);
		};
	}, [state.notice, studio]);
	const problem = configurationProblem(state.settings, state.session);
	const controlsDisabled = state.processing || state.initializing;
	const processDisabled =
		controlsDisabled ||
		state.uploading ||
		state.refreshing ||
		!state.source ||
		!state.session?.ready ||
		Boolean(problem);
	const exportDisabled =
		state.initializing ||
		state.uploading ||
		state.restoring ||
		exporting ||
		!(state.mode === 'original' ? state.source : state.result);
	const progress = state.progress;
	const elapsed = Math.floor(progress?.elapsed ?? 0);
	const handleOpen = () => {
		fileInput.current?.click();
	};
	const handleUpload = (file: File) => {
		void studio.upload(file);
	};
	const handleFile = (event: ChangeEvent<HTMLInputElement>) => {
		const file = event.currentTarget.files?.[0];
		if (file) handleUpload(file);
		event.currentTarget.value = '';
	};
	const handleSettings = () => {
		try {
			setSettingsValue(studio.validatedSettings());
		} catch (error) {
			studio.notify(errorMessage(error), true);
		}
	};
	const handleCloseSettings = () => setSettingsValue(null);
	const handleSaveSettings = (value: SettingsT) => {
		studio.applySettings(value);
		setSettingsValue(null);
		studio.notify('Settings saved');
	};
	const handleRefresh = () => studio.refresh();
	const handleHelp = () => setHelpOpen(true);
	const handleCloseHelp = () => setHelpOpen(false);
	const handleProcess = () => {
		void studio.process();
	};
	const handleExport = async () => {
		if (exportDisabled) return;
		setExporting(true);
		try {
			await viewer.current?.exportImage();
		} catch (error) {
			studio.notify(errorMessage(error), true);
		} finally {
			setExporting(false);
		}
	};
	return (
		<>
			<input ref={fileInput} type="file" accept="image/*" hidden onChange={handleFile} />
			<Shell>
				<Editor aria-label="Image editor">
					<Viewer
						ref={viewer}
						source={state.source}
						result={state.result}
						mode={state.mode}
						exportRevision={state.exportRevision}
						onModeChange={studio.setMode}
						metrics={state.metrics}
						onOpen={handleOpen}
						onSettings={handleSettings}
						onHelp={handleHelp}
						onUpload={handleUpload}
						uploadDisabled={controlsDisabled || state.uploading}
						settingsDisabled={controlsDisabled}
						onError={studio.notify}
					/>
				</Editor>
				<Sidebar aria-label="Processing parameters">
					<SettingsScroll>
						<GenerationPanel
							settings={state.settings}
							seed={state.seed}
							disabled={controlsDisabled}
							onChange={studio.changeSettings}
							onSeed={studio.setSeed}
							onReset={studio.reset}
						/>
						<RestorationPanel settings={state.settings} disabled={controlsDisabled} onChange={studio.changeSettings} />
					</SettingsScroll>
					<ProcessPanel>
						{(state.processing || state.restoring) && (
							<Feedback role="status" aria-live="polite">
								<div>
									<span>
										{state.restoring && !state.processing
											? 'Updating detail restoration…'
											: progress?.stage || 'Processing…'}
									</span>
									{progress && (
										<span>
											{Math.round(progress.progress ?? 0)}% · {Math.floor(elapsed / 60)}:
											{String(elapsed % 60).padStart(2, '0')}
										</span>
									)}
								</div>
								<progress
									max={100}
									{...(progress?.progress == null ? {} : { value: progress.progress })}
									aria-label="Processing progress"
								/>
							</Feedback>
						)}
						<Actions>
							<Button
								type="button"
								$primary
								disabled={processDisabled}
								title={problem || 'Process image'}
								onClick={handleProcess}
							>
								<Icon name="play" />
								{state.processing ? 'Processing…' : 'Process image'}
							</Button>
							<Button type="button" disabled={exportDisabled} onClick={handleExport}>
								<Icon name="download" />
								{exporting ? 'Exporting…' : 'Export PNG'}
							</Button>
						</Actions>
					</ProcessPanel>
				</Sidebar>
			</Shell>
			{settingsValue && (
				<Settings
					value={settingsValue}
					session={state.session}
					refreshing={state.refreshing}
					onRefresh={handleRefresh}
					onSave={handleSaveSettings}
					onClose={handleCloseSettings}
				/>
			)}
			{helpOpen && <HelpDialog onClose={handleCloseHelp} />}
			{state.notice && (
				<Toast $error={state.notice.error} role={state.notice.error ? 'alert' : 'status'}>
					{state.notice.message}
				</Toast>
			)}
		</>
	);
};

export default ImageEditor;
