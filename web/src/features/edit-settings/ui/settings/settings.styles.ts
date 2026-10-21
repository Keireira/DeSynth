import styled from 'styled-components';
import { Button } from '@ui';

export const Dialog = styled.dialog`
	width: calc(100% - 32px);
	max-width: 860px;
	max-height: calc(100dvh - 48px);
	padding: 0;
	border: 0;
	border-radius: 8px;
	background: #242424;
	color: #e8e8e8;
	font: inherit;
	&[open] {
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}
	&::backdrop {
		background: #00000099;
	}
`;

export const Heading = styled.header`
	display: flex;
	align-items: center;
	justify-content: space-between;
	flex-shrink: 0;
	gap: 16px;
	padding: 20px 24px;
	h2 {
		margin: 0;
		font-size: 18px;
		font-weight: 600;
	}
`;

export const HeadingActions = styled.div`
	display: flex;
	align-items: center;
	gap: 8px;
	button {
		height: 32px;
	}
`;

export const Content = styled.div`
	display: grid;
	grid-template-columns: minmax(220px, 0.85fr) minmax(0, 1.5fr);
	align-items: start;
	gap: 32px;
	min-height: 0;
	padding: 0 24px;
	overflow: auto;
	scrollbar-width: thin;
	scrollbar-color: #666666 transparent;
	@media (max-width: 650px) {
		grid-template-columns: minmax(0, 1fr);
		gap: 28px;
	}
`;

export const Actions = styled.footer`
	display: flex;
	justify-content: flex-end;
	flex-shrink: 0;
	gap: 8px;
	padding: 18px 24px;
	background: #242424;
	${Button} {
		padding: 0 18px;
	}
`;

export const SettingsColumn = styled.div`
	display: flex;
	flex-direction: column;
	gap: 20px;
	min-width: 0;
`;

export const Message = styled.p`
	margin: 0;
	color: var(--muted);
	font-size: 13px;
	line-height: 1.5;
	overflow-wrap: anywhere;
`;

export const Feedback = styled(Message)`
	flex-shrink: 0;
	margin: 16px 24px 0;
	color: #e8e8e8;
`;
