default:
	pip install .
	-rm -rf dist build gunpowder.egg-info

.PHONY: install-full
install-full:
	pip install .[full]

.PHONY: install-dev
install-dev:
	pip install -e .[full]

.PHONY: test
test:
	python -m tests -v

.PHONY: publish
publish:
	-rm -rf dist build gunpowder.egg-info
	python setup.py sdist bdist_wheel
	twine upload dist/*
	-rm -rf dist build gunpowder.egg-info
