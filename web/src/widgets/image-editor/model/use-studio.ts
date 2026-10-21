import { useEffect, useState, useSyncExternalStore } from 'react';
import { createStudio } from './studio';

export const useStudio = () => {
	const [studio] = useState(createStudio);
	const state = useSyncExternalStore(studio.subscribe, studio.getSnapshot);
	useEffect(() => {
		const stop = studio.start();
		return stop;
	}, [studio]);
	return { state, studio };
};
