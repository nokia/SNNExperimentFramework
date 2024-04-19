.PHONY: clean virtualenv test install

PYTHON_PATH := $(shell which python3)

test:
		coverage run

clean:
		find . -name '*.py[co]' -delete

virtualenv:
		virtualenv --python="$(PYTHON_PATH)" --prompt="|> D5G <|" env
		env/bin/pip3 install -r requirements.txt
		env/bin/pip3 freeze | sed -ne 's/==.*//p' | xargs env/bin/pip3 install -U
		@echo
		@echo "Virtualenv created, use 'source src/env/bin/activate' to use it"
		@echo

install:
		sudo apt-get -y install python-virtualenv python3-dev
