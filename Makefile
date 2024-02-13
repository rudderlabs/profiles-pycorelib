PRODUCT_NAME=profiles-pycorelib

VERS=0.11.0rc1
GIT_COMMIT=`git rev-parse --short HEAD`
VERS_DATE=`date -u +%Y-%m-%d\ %H:%M`
VERS_FILE=./version.py

update_version:
	/bin/rm -f $(VERS_FILE)
	@echo "# WARNING: auto-generated by Makefile release target -- run 'make release' to update" > $(VERS_FILE)
	@echo "" >> $(VERS_FILE)
	@echo "" >> $(VERS_FILE)
	@echo "version = \"$(VERS)\"" >> $(VERS_FILE)
	@echo "git_commit = \"$(GIT_COMMIT)\"  # the commit JUST BEFORE the release" >> $(VERS_FILE)
	@echo "version_date = \"$(VERS_DATE)\"  # UTC" >> $(VERS_FILE)
	/bin/cat $(VERS_FILE)

check_git:
	@if [ "a" != "a$$(git status --untracked-files=no --porcelain)" ]; \
	then \
		echo "There are local changes. git status should be clean."; \
		false; \
	fi
	@if [ "a" != "a$$(git tag -l $(VERS))" ]; \
	then \
		echo Tag $(VERS) already exists. Remove this tag and retry, if you are sure.; \
		false; \
	fi
	@echo "Are you sure you want to make release version $(VERS)? The command will update setup.py and commit and push new git tag. [y/N] " && read ans && [ $${ans:-N} = y ]

# This is required for release process. make release updates the version and creates the tag for release
release: check_git update_version
	git commit -am "$(VERS) release"
	git tag -a $(VERS) -m "$(VERS) release"
	git push
	git push origin --tags

install:
	SKIP_PB_BIN=true pip3 install .

.PHONY: update_version check_git release install