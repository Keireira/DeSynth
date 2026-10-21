import styled from 'styled-components';

export const Button = styled.button<{ $primary?: boolean }>`
	display: inline-flex;
	align-items: center;
	justify-content: center;
	gap: 6px;
	height: 39px;
	min-width: 0;
	padding: 0 12px;
	border: 0;
	border-radius: 4px;
	background: ${({ $primary }) => ($primary ? 'var(--accent)' : '#303030')};
	color: ${({ $primary }) => ($primary ? '#181818' : '#eeeeee')};
	font-size: 13px;
	font-weight: ${({ $primary }) => ($primary ? 600 : 400)};
	white-space: nowrap;
	&:hover:not(:disabled) {
		background: ${({ $primary }) => ($primary ? '#ffb09f' : '#383838')};
	}
`;
export const IconButton = styled.button`
	display: inline-grid;
	place-items: center;
	flex-shrink: 0;
	width: 28px;
	height: 28px;
	padding: 4px;
	border: 0;
	border-radius: 4px;
	background: transparent;
	color: inherit;
	&:hover:not(:disabled) {
		background: #383838;
	}
`;
export const FieldLabel = styled.label`
	display: block;
	margin-bottom: 7px;
	color: var(--muted);
	font-size: 13px;
`;
export const Select = styled.select`
	width: 100%;
	height: 38px;
	min-width: 0;
	padding: 0 9px;
	border: 0;
	border-radius: 4px;
	background: #181818;
	color: #e8e8e8;
	text-overflow: ellipsis;
	cursor: pointer;
`;
export const TextInput = styled.input`
	width: 100%;
	height: 38px;
	min-width: 0;
	padding: 0 10px;
	border: 0;
	border-radius: 4px;
	background: #181818;
	color: #e8e8e8;
	&::placeholder {
		color: #aaaaaa;
	}
`;
