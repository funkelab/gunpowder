default:
	pip install .
	-rm -rf dist build gunpowder.egg-info

.PHONY: install-full
install-full:
	pip install .[full]

.PHONY: install-dev
install-dev:
	pip install -r requirements-dev.txt
	pip install -e .[full]

.PHONY: test
test:
	pytest -v --cov gunpowder

.PHONY: publish
publish:
	-rm -rf dist build gunpowder.egg-info
	python setup.py sdist bdist_wheel
	twine upload dist/*
	-rm -rf dist build gunpowder.egg-info
