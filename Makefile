VERS?=0.11.0
VERS_FILE=./version.py

.PHONY: update_version
update_version:
	/bin/rm -f $(VERS_FILE)
	@echo "" >> $(VERS_FILE)
	@echo "version = \"$(VERS)\"" >> $(VERS_FILE)
	/bin/cat $(VERS_FILE)

.PHONY: check_git
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


# TODO: this needs to be removed after the new release process is established
.PHONY: release
release: check_git update_version
	git commit -am "v$(VERS) release"
	git tag -a v$(VERS) -m "v$(VERS) release"
	git push
	git push origin --tags

.PHONY: install
install:
	SKIP_PB_BIN=true pip3 install .
