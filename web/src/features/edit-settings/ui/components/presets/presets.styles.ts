import styled from 'styled-components';
import { Button } from '../../../../../shared/ui';

export const Section = styled.section`
	display: flex;
	flex-direction: column;
	gap: 12px;
	min-width: 0;
`;

export const Actions = styled.div`
	display: flex;
	gap: 8px;
	${Button} {
		height: 34px;
	}
	${Button}:first-child {
		flex: 1;
	}
	${Button}:last-child {
		padding: 0 16px;
	}
`;
