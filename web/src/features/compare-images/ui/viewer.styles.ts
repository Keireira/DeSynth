import styled from 'styled-components';

export const Root = styled.section`
	container-type: inline-size;
	flex: 1;
	min-width: 0;
	min-height: 0;
	display: flex;
	flex-direction: column;
	overflow: hidden;
	position: relative;
`;
export const Viewport = styled.div<{ $dragging: boolean }>`
	position: relative;
	flex: 1;
	min-height: 200px;
	overflow: hidden;
	touch-action: none;
	background: #181818;
	cursor: ${({ $dragging }) => ($dragging ? 'grabbing' : 'grab')};
`;
export const Canvas = styled.canvas`
	position: absolute;
	inset: 0;
	display: block;
	width: 100%;
	height: 100%;
`;
export const EmptyState = styled.div`
	position: absolute;
	inset: 0;
	display: flex;
	flex-direction: column;
	align-items: center;
	justify-content: center;
	gap: 10px;
	pointer-events: none;
	background: #181818;
	color: #aaa;
	svg {
		width: 35px;
		height: 35px;
		margin-bottom: 9px;
	}
	strong {
		font-size: 16px;
		font-weight: 500;
	}
	p,
	span {
		font-size: 13px;
	}
`;
export const CanvasLabel = styled.span<{ $right?: boolean }>`
	position: absolute;
	top: 14px;
	left: ${({ $right }) => ($right ? 'auto' : '14px')};
	right: ${({ $right }) => ($right ? '14px' : 'auto')};
	z-index: 1;
	padding: 5px 10px;
	background: #242424e6;
	font-size: 13px;
	color: #eee;
	pointer-events: none;
	border-radius: 3px;
`;
export const Divider = styled.div`
	position: absolute;
	width: 2px;
	background: #eee;
	pointer-events: none;
`;
export const DividerHandle = styled.button`
	position: absolute;
	left: 50%;
	top: 50%;
	transform: translate(-50%, -50%);
	display: grid;
	place-items: center;
	width: 44px;
	height: 44px;
	border: 2px solid #eee;
	border-radius: 4px;
	background: #242424e6;
	color: #eee;
	padding: 9px;
	pointer-events: auto;
	cursor: ew-resize;
	svg {
		width: 24px;
		height: 24px;
		fill: none;
		stroke: currentColor;
		stroke-width: 1.5;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
`;
export const DropOverlay = styled.div`
	position: absolute;
	inset: 8px;
	z-index: 4;
	display: flex;
	align-items: center;
	justify-content: center;
	background: #242424e6;
	border: 1px dashed #eee;
	color: #eee;
	font-size: 15px;
	pointer-events: none;
`;
