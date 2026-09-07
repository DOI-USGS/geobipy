# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = ../docs
# Sphinx's pickled parse cache. It only speeds up the *next* build and must not
# be published, so it is kept outside BUILDDIR (which is what gets served by
# GitHub Pages). Without an explicit -d it defaults to $(BUILDDIR)/.doctrees.
DOCTREESDIR   = ../.doctrees

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help html Makefile

# The html lands at the top of ../docs/ because GitHub Pages can only serve a
# branch's root or its /docs folder -- there is no setting that points at a
# nested subdirectory. -d keeps the parse cache out of that published tree.
# The mpi_gallery step is repeated from the catch-all below: an explicit target
# takes precedence over the pattern rule, so it would otherwise be skipped.
html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" -d "$(DOCTREESDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -b $@ "$(SOURCEDIR)" "$(BUILDDIR)" -d "$(DOCTREESDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"
	rm -rf "$(DOCTREESDIR)"
	rm -rf ./examples
	rm -f file_hashes.json
