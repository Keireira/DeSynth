.PHONY: web-install web-build web-dev web-check web-format

# One workspace lockfile and one copy of the formatter/linter live at the root.
web-install:
	$(PNPM) install --frozen-lockfile

web-build: web-install
	$(PNPM) build

web-dev: web-install
	$(NODE) scripts/web-dev.mjs --backend-port "$(PORT)" --port "$(WEB_PORT)" --pnpm "$(PNPM)"

web-check: web-install
	$(PNPM) check

web-format: web-install
	$(PNPM) format

setup: web-build
