import { useEffect, useId, useRef, useState, type MouseEvent, type SyntheticEvent } from 'react';
import type { SettingsT } from '@entities/editor-session';
import { Button, IconButton, Icon } from '@ui';
import { Models, Presets, Runtime } from './components';
import { configurationProblem, migrateModelPaths, normalizeSettings, persistSettings } from '@entities/editor-session';
import type { NotificationT, Props } from './settings.d';
import {
	Actions,
	Content,
	Dialog,
	Feedback,
	Heading,
	HeadingActions,
	Message,
	SettingsColumn
} from './settings.styles';

const Settings = ({ value, session, refreshing, onRefresh, onSave, onClose }: Props) => {
	const [draft, setDraft] = useState(() => normalizeSettings(value));
	const [notification, setNotification] = useState<NotificationT | null>(null);
	const dialogRef = useRef<HTMLDialogElement>(null);
	const titleId = useId();

	useEffect(() => {
		const dialog = dialogRef.current;
		dialog?.showModal();

		return () => {
			dialog?.close();
		};
	}, []);

	const availableDraft = session ? migrateModelPaths(draft, session) : draft;
	const problem = configurationProblem(availableDraft, session);
	const saveDisabled = refreshing || Boolean(problem);

	const handleNotification = (message: string, error = false) => {
		setNotification({ message, error });
	};

	const handleChange = (settings: SettingsT) => {
		setNotification(null);
		setDraft(settings);
	};

	const handleRefresh = async () => {
		setNotification(null);

		try {
			await onRefresh();
		} catch (error) {
			handleNotification(error instanceof Error ? error.message : 'Could not refresh models and devices.', true);
		}
	};

	const handleSave = () => {
		if (saveDisabled) {
			return;
		}

		try {
			const saved = normalizeSettings(availableDraft);
			persistSettings(saved);
			onSave(saved);
		} catch (error) {
			handleNotification(error instanceof Error ? error.message : 'Could not save settings.', true);
		}
	};

	const handleCancel = (event: SyntheticEvent<HTMLDialogElement>) => {
		event.preventDefault();
		onClose();
	};

	const handleBackdrop = (event: MouseEvent<HTMLDialogElement>) => {
		if (event.target !== event.currentTarget) {
			return;
		}

		const bounds = event.currentTarget.getBoundingClientRect();
		const outside =
			event.clientX < bounds.left ||
			event.clientX > bounds.right ||
			event.clientY < bounds.top ||
			event.clientY > bounds.bottom;

		if (outside) {
			onClose();
		}
	};

	return (
		<Dialog ref={dialogRef} aria-labelledby={titleId} onCancel={handleCancel} onClick={handleBackdrop}>
			<Heading>
				<h2 id={titleId}>Settings</h2>
				<HeadingActions>
					<Button
						type="button"
						disabled={refreshing}
						title="Rescan available models and devices"
						onClick={handleRefresh}
					>
						{refreshing ? 'Refreshing…' : 'Refresh'}
					</Button>
					<IconButton type="button" aria-label="Close settings without saving" onClick={onClose}>
						<Icon name="close" />
					</IconButton>
				</HeadingActions>
			</Heading>
			<Content>
				<Presets
					value={availableDraft}
					session={session}
					disabled={refreshing}
					onChange={handleChange}
					onError={handleNotification}
				/>
				<SettingsColumn>
					<Models value={availableDraft} session={session} disabled={refreshing} onChange={handleChange} />
					<Runtime value={availableDraft} session={session} disabled={refreshing} onChange={handleChange} />
					{problem && (
						<Message role="status" aria-live="polite">
							{problem}
						</Message>
					)}
				</SettingsColumn>
			</Content>
			{notification && <Feedback role={notification.error ? 'alert' : 'status'}>{notification.message}</Feedback>}
			<Actions>
				<Button type="button" onClick={onClose}>
					Cancel
				</Button>
				<Button type="button" $primary disabled={saveDisabled} onClick={handleSave}>
					Save settings
				</Button>
			</Actions>
		</Dialog>
	);
};

export default Settings;
