import styled from 'styled-components';

export const Toggle = styled.label`
	position: relative;
	width: 32px;
	height: 19px;
	cursor: pointer;
	input {
		position: absolute;
		opacity: 0;
		inset: 0;
		width: 100%;
		height: 100%;
		margin: 0;
	}
	span {
		display: block;
		height: 100%;
		background: #242424;
		border: 2px solid #666;
	}
	span::after {
		content: '';
		position: absolute;
		top: 5px;
		left: 5px;
		width: 9px;
		height: 9px;
		background: #666;
	}
	input:checked + span {
		background: #e8e8e8;
		border-color: #e8e8e8;
	}
	input:checked + span::after {
		transform: translateX(13px);
		background: #242424;
	}
	input:focus-visible + span {
		outline: 2px solid #e8e8e8;
		outline-offset: 3px;
	}
`;
export const RestoreFields = styled.fieldset`
	min-width: 0;
	margin: 0;
	padding: 0;
	border: 0;
	&:disabled {
		opacity: 0.35;
	}
	&:disabled input,
	&:disabled select {
		opacity: 1;
	}
`;
export const Sharpen = styled.div`
	margin-top: 20px;
`;
export const Check = styled.label`
	display: flex;
	align-items: center;
	gap: 8px;
	margin-top: 20px;
	font-size: 13px;
	color: var(--muted);
	cursor: pointer;
	input {
		width: 12px;
		height: 12px;
		margin: 0;
		accent-color: #e8e8e8;
	}
`;
