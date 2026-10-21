import styled from 'styled-components';

export const Dialog = styled.dialog`
	width: calc(100% - 32px);
	max-width: 540px;
	max-height: calc(100dvh - 48px);
	overflow: auto;
	padding: 24px;
	border: 0;
	border-radius: 8px;
	background: var(--panel);
	color: #e8e8e8;
	&::backdrop {
		background: #0009;
	}
	header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 20px;
	}
	h2 {
		font-size: 18px;
		font-weight: 600;
	}
	dl {
		margin: 0;
	}
	dt {
		margin: 17px 0 5px;
		font-size: 13px;
		font-weight: 600;
	}
	dd {
		margin: 0;
		color: var(--muted);
		font-size: 13px;
		line-height: 1.7;
	}
`;
