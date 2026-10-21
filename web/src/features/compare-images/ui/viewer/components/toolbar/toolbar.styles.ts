import styled from 'styled-components';

export const Root = styled.div`
	display: flex;
	flex-wrap: nowrap;
	align-items: center;
	gap: 12px;
	min-width: 0;
	max-width: 100%;
	flex-shrink: 0;
	overflow: hidden;
	padding: 10px 12px;
	background: #242424;
	@media (max-width: 1100px) {
		gap: 8px;
	}
`;
export const ScrollGroup = styled.div`
	display: flex;
	flex: 1 1 auto;
	align-items: center;
	gap: 12px;
	min-width: 0;
	overflow-x: auto;
	overflow-y: hidden;
	scrollbar-width: thin;
	scrollbar-color: #666 transparent;
	@media (max-width: 1100px) {
		gap: 8px;
	}
`;
export const FixedTools = styled.div`
	display: flex;
	flex: 0 0 auto;
	align-items: center;
	gap: 8px;
	@media (max-width: 1100px) {
		gap: 4px;
	}
`;
export const TextButton = styled.button`
	display: inline-flex;
	align-items: center;
	justify-content: center;
	gap: 7px;
	height: 36px;
	flex: 0 0 auto;
	padding: 0 12px;
	border: 0;
	border-radius: 4px;
	background: #303030;
	color: #e8e8e8;
	font-size: 13px;
	white-space: nowrap;
	svg {
		width: 16px;
		height: 16px;
	}
	&:hover:not(:disabled) {
		background: #383838;
	}
`;
export const ModeTabs = styled.div`
	display: flex;
	flex-wrap: nowrap;
	align-items: center;
	gap: 3px;
	flex: 0 0 auto;
	min-width: max-content;
	@media (max-width: 1100px) {
		gap: 1px;
	}
`;
export const ModeButton = styled.button<{ $active: boolean }>`
	padding: 0 13px;
	background: ${({ $active }) => ($active ? '#ff9b85' : 'transparent')};
	color: ${({ $active }) => ($active ? '#181818' : '#aaa')};
	font-size: 13px;
	font-weight: ${({ $active }) => ($active ? 600 : 400)};
	white-space: nowrap;
	border: 0;
	border-radius: 4px;
	height: 34px;
	&:hover:not(:disabled) {
		filter: brightness(1.1);
		color: ${({ $active }) => ($active ? '#181818' : '#eee')};
	}
	@media (max-width: 1300px) {
		padding: 0 11px;
	}
	@media (max-width: 1100px) {
		padding: 0 8px;
	}
`;
export const Metrics = styled.div`
	display: flex;
	flex: 1 0 auto;
	align-items: center;
	justify-content: flex-end;
	gap: 12px;
	span {
		display: flex;
		flex-direction: column;
		gap: 3px;
		font-size: 12px;
		color: #aaa;
		white-space: nowrap;
	}
	strong {
		font-size: 13px;
		color: #e8e8e8;
		font-weight: 400;
		font-variant-numeric: tabular-nums;
	}
	@container (max-width: 899px) {
		display: none;
	}
`;
export const ZoomControls = styled.div`
	display: flex;
	align-items: center;
	gap: 1px;
	flex-shrink: 0;
	margin-left: auto;
	span {
		min-width: 39px;
		text-align: center;
		font-size: 13px;
		font-variant-numeric: tabular-nums;
	}
`;
export const ToolButton = styled.button`
	display: inline-grid;
	place-items: center;
	width: 28px;
	height: 28px;
	flex: 0 0 28px;
	padding: 4px;
	background: transparent;
	color: #e8e8e8;
	border: 0;
	border-radius: 4px;
	svg {
		width: 16px;
		height: 16px;
	}
	&:hover:not(:disabled) {
		background: #303030;
	}
`;
export const DiffControls = styled.div`
	display: flex;
	flex: 0 0 auto;
	align-items: center;
	gap: 10px;
	label {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 13px;
		color: #aaa;
		white-space: nowrap;
	}
	input {
		width: 76px;
		accent-color: #ff9b85;
	}
	output {
		min-width: 24px;
		color: #e8e8e8;
		font-variant-numeric: tabular-nums;
	}
	select {
		width: 150px;
		height: 34px;
		padding: 0 8px;
		border: 0;
		border-radius: 4px;
		background: #181818;
		color: #e8e8e8;
		font-size: 13px;
	}
`;
