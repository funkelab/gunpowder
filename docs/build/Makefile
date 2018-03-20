# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = gunpowder
SOURCEDIR     = .
BUILDDIR      = _build

default:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	-rm ../*.html
	-rm ../*.inv
	-rm ../*.js
	-rm -r ../_sources
	-rm -r ../_modules
	-rm -r ../_static
	-rm -r ../_downloads
	-rm -r ../_images
	cp -r _build/html/* ..
	-rm -r ../_themes

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
