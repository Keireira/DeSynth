import styled from 'styled-components';

export const Shell = styled.main`
	display: grid;
	grid-template-columns: minmax(0, 1fr) 352px;
	gap: 14px;
	height: 100dvh;
	min-height: 720px;
	@media (max-width: 1300px) {
		grid-template-columns: minmax(0, 1fr) 320px;
		gap: 12px;
	}
	@media (max-width: 1100px) {
		grid-template-columns: minmax(0, 1fr) 300px;
	}
	@media (max-width: 850px) {
		grid-template-columns: minmax(0, 1fr);
		height: auto;
		min-height: 100dvh;
	}
`;
export const Editor = styled.section`
	display: flex;
	min-width: 0;
	min-height: 0;
	@media (max-width: 850px) {
		height: 65vh;
		min-height: 430px;
	}
`;
export const Sidebar = styled.aside`
	display: flex;
	flex-direction: column;
	gap: 12px;
	min-width: 0;
	min-height: 0;
`;
export const SettingsScroll = styled.div`
	display: flex;
	flex: 1;
	flex-direction: column;
	gap: 12px;
	min-height: 0;
	overflow: auto;
	scrollbar-width: thin;
	scrollbar-color: #666 transparent;
`;
export const ProcessPanel = styled.div`
	flex-shrink: 0;
	padding: 16px 18px;
	border-radius: 2px;
	background: var(--panel);
	@media (max-width: 1100px) {
		padding: 14px;
	}
`;
export const Feedback = styled.div`
	font-size: 13px;
	color: var(--muted);
	div {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 8px;
	}
	span:first-child {
		min-width: 0;
		overflow-wrap: anywhere;
	}
	span:last-child {
		white-space: nowrap;
		font-variant-numeric: tabular-nums;
	}
	progress {
		display: block;
		appearance: none;
		width: 100%;
		height: 3px;
		margin: 10px 0 15px;
		border: 0;
		background: #666;
	}
	progress::-webkit-progress-bar {
		background: #666;
	}
	progress::-webkit-progress-value {
		background: #e8e8e8;
	}
	progress::-moz-progress-bar {
		background: #e8e8e8;
	}
`;
export const Actions = styled.div`
	display: grid;
	grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
	gap: 8px;
	button {
		padding: 0 7px;
	}
	svg {
		width: 15px;
		height: 15px;
	}
`;
export const Toast = styled.div<{ $error: boolean }>`
	position: fixed;
	bottom: 26px;
	left: 50%;
	transform: translateX(-50%);
	z-index: 50;
	max-width: calc(100% - 32px);
	padding: 12px 18px;
	border: 1px solid ${({ $error }) => ($error ? '#e8e8e8' : '#666')};
	border-radius: 2px;
	background: var(--panel);
	color: #eee;
	font-size: 13px;
`;
