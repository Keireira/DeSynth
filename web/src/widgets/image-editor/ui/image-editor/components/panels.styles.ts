import styled from 'styled-components';

export const Section = styled.section`
	padding: 19px 20px;
	border: 0;
	border-radius: 2px;
	background: var(--panel);
	@media (max-width: 1300px) {
		padding: 18px;
	}
	@media (max-width: 1100px) {
		padding: 17px;
	}
`;
export const Heading = styled.div`
	display: flex;
	align-items: center;
	justify-content: space-between;
	gap: 12px;
	margin-bottom: 21px;
	h2 {
		font-size: 16px;
		font-weight: 600;
	}
`;
export const FieldRow = styled.div`
	display: grid;
	grid-template-columns: 65px minmax(0, 1fr);
	align-items: center;
	gap: 12px;
	label {
		margin: 0;
	}
`;
