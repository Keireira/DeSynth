import { createGlobalStyle } from 'styled-components';

const GlobalStyle = createGlobalStyle`
	:root {
		font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI Variable Text', 'Segoe UI', sans-serif;
		font-synthesis: none;
		font-size: 14px;
		line-height: 1.45;
		font-weight: 400;
		color-scheme: dark;
		color: #e8e8e8;
		background: #141414;
		--accent: #ff9b85;
		--panel: #242424;
		--muted: #aaaaaa;
	}
	* { box-sizing: border-box; }
	body { margin: 0; min-width: 340px; }
	button, input, select { font: inherit; }
	button { cursor: pointer; color: inherit; }
	button, input, select { outline: none; }
	button:disabled, input:disabled, select:disabled { opacity: .35; cursor: not-allowed; }
	:focus-visible { outline: 2px solid #e8e8e8; outline-offset: 3px; }
	[hidden] { display: none !important; }
	h1, h2, h3, p { margin: 0; }
	@media (prefers-reduced-motion: reduce) {
		*, *::before, *::after { animation: none !important; transition: none !important; }
	}
`;

export default GlobalStyle;
