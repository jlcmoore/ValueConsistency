test:
	cd src/ && ../env-valueconsistency/bin/python -m unittest tests

init:
	python3.11 -m venv env-valueconsistency
	env-valueconsistency/bin/pip install -r requirements.txt
	env-valueconsistency/bin/pip install --editable .
	env-valueconsistency/bin/python -m ipykernel install --user --name "env-valueconsistency"

.PHONY: init test
