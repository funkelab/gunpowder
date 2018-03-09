default:
	python setup.py install
	-rm -rf dist build gunpowder.egg-info

.PHONY: dev
dev:
	pip install -e .

.PHONY: test
test:
	python tests/test_suite.py
