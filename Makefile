PRODUCT_NAME=profiles-pycorelib

VERS=0.11.0rc1
GIT_COMMIT=git rev-parse --short HEAD
VERS_DATE=date -u +%Y-%m-%d\ %H:%M
PYCORELIB_VERS_FILE=./setup.py

update_version:
	sed -i '' 's/version = .*/version = "$(VERS)"/' $(PYCORELIB_VERS_FILE)
	sed -i '' 's/git_commit = .*/git_commit = "$(shell $(GIT_COMMIT))" # the commit JUST BEFORE the release /' $(PYCORELIB_VERS_FILE)
	sed -i '' 's/version_date = .*/version_date = "$(shell $(VERS_DATE))" # UTC/' $(PYCORELIB_VERS_FILE)

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