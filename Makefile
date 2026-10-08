.PHONY: up down logs migrate seed test backup restore

# One-command staging stack (docs/adr/0014/0018, docs/DEPLOYMENT.md).
# `backup`/`restore` require TAG=<name>, e.g. `make backup TAG=2026-10-08`.

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

migrate:
	docker compose run --rm migrate

seed:
	docker compose exec api python scripts/seed_staging.py

test:
	cd server && pytest -q
	pytest -q

backup:
	@if [ -z "$(TAG)" ]; then echo "usage: make backup TAG=<name>" >&2; exit 1; fi
	docker compose exec api python scripts/backup.py $(TAG)

restore:
	@if [ -z "$(TAG)" ]; then echo "usage: make restore TAG=<name>" >&2; exit 1; fi
	docker compose exec api python scripts/restore.py $(TAG)
