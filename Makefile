PY ?= python
.PHONY: fetch dictionary llm evaluate harmonize indicators report all test lint
fetch:      ; $(PY) -m enahoharm.pipeline fetch
dictionary: ; $(PY) -m enahoharm.pipeline dictionary
llm:        ; $(PY) -m enahoharm.pipeline llm
evaluate:   ; $(PY) -m enahoharm.pipeline evaluate
harmonize:  ; $(PY) -m enahoharm.pipeline harmonize
indicators: ; $(PY) -m enahoharm.pipeline indicators
report:     ; $(PY) -m enahoharm.pipeline report
all:        ; $(PY) -m enahoharm.pipeline all
test:       ; $(PY) -m pytest -q
lint:       ; ruff check src tests
