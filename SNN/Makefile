# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

.PHONY: clean virtualenv test install docs pylint

PYTHON_PATH := $(shell which python3)

test:
		coverage run

pylint:
		find SNN2/ -type f -name "*.py" | xargs pylint > pylint_output.txt

clean:
		find . -name '*.py[co]' -delete

virtualenv:
		virtualenv --python="$(PYTHON_PATH)" --pip 25.2 --prompt="|> SNN <|" env
		env/bin/pip install --upgrade pip
		grep -v '^\-e' requirements.txt | cut -d = -f 1 | xargs -n1 env/bin/pip3 install -U
		@echo
		@echo "Virtualenv created, use 'source src/env/bin/activate' to use it"
		@echo

install:
		sudo apt-get -y install python-virtualenv python3-dev

docs:
	sphinx-apidoc -f -e -o docs/source SNN2
	sphinx-build -b html docs/source docs/public