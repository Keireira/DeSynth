import type { Props } from './icon.d';
import { Svg } from './icon.styles';

const paths = {
	image: 'M3 3h18v18H3ZM3 17l5-5 4 4 4-6 5 7M8 8h.01',
	plus: 'M12 5v14M5 12h14',
	minus: 'M5 12h14',
	fit: 'M4 9V4h5m6 0h5v5m0 6v5h-5m-6 0H4v-5',
	help: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM9.5 9a2.5 2.5 0 0 1 5 .5c0 1.5-2.5 2-2.5 3.5m0 3h.01',
	close: 'm6 6 12 12M6 18 18 6',
	download: 'M12 4v12m-4-4 4 4 4-4M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3',
	reset: 'M4 10a8 8 0 1 1 1 8M4 4v6h6',
	dice: 'M6 4h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2ZM8 8h.01M16 8h.01M12 12h.01M8 16h.01M16 16h.01',
	play: 'm8 4 12 8-12 8Z'
};

const Icon = ({ name, className }: Props) => (
	<Svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
		<path d={paths[name]} />
	</Svg>
);

export default Icon;
