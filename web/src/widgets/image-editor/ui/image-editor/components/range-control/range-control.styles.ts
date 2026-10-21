import styled from 'styled-components';

export const Root = styled.div`
	margin-bottom: 21px;
	&:last-child {
		margin-bottom: 0;
	}
`;
export const Caption = styled.div`
	display: flex;
	justify-content: space-between;
	align-items: center;
	gap: 12px;
	margin-bottom: 11px;
	font-size: 13px;
	label {
		color: var(--muted);
	}
	output {
		font-variant-numeric: tabular-nums;
	}
`;
export const Slider = styled.input<{ $progress: number }>`
	appearance: none;
	display: block;
	width: 100%;
	height: 4px;
	margin: 0;
	border: 0;
	border-radius: 2px;
	cursor: pointer;
	background: linear-gradient(
		to right,
		var(--accent) 0%,
		var(--accent) ${({ $progress }) => $progress}%, #666 ${({ $progress }) => $progress}%, #666 100%
	);
	&::-webkit-slider-thumb {
		appearance: none;
		width: 13px;
		height: 13px;
		border: 0;
		border-radius: 2px;
		background: var(--accent);
	}
	&::-moz-range-thumb {
		width: 13px;
		height: 13px;
		border: 0;
		border-radius: 2px;
		background: var(--accent);
	}
`;
