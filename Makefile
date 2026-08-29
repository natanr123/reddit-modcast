PY := .venv/bin/python

.PHONY: test data ingest index induce eval eval-llm predict-demo

test:
	.venv/bin/pytest -q

data:
	$(PY) scripts/make_dataset.py

ingest:
	$(PY) -m modcast.cli ingest

index:
	$(PY) -m modcast.cli index

induce:
	$(PY) -m modcast.cli induce

eval:
	$(PY) -m modcast.cli eval

eval-llm:
	$(PY) -m modcast.cli eval --with-llm

predict-demo:
	$(PY) -m modcast.cli predict --sub legaladvice \
		--title "Landlord kept my deposit, what can I do?" \
		--body "My landlord in Ohio is keeping my $$1200 deposit for 'cleaning' but the place was spotless. He never sent an itemized list. What are my options?"
