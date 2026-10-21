import styled from 'styled-components';
import { TextInput } from '../../../../shared/ui';

export const SeedField = styled.div`
	display: flex;
	align-items: center;
	min-width: 0;
	padding-right: 3px;
	border-radius: 4px;
	background: #181818;
`;
export const SeedInput = styled(TextInput)`
	height: 34px;
	font-size: 13px;
`;
