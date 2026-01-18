COPY_FILES = $(patsubst src/%,public/%,$(wildcard src/*) $(wildcard src/*/*))

all: $(COPY_FILES)

public/%: src/%
	@echo $@
	@rm -rf $@
	@cp -r $< $@

deploy:
	rsync -a --info=name src/ public/ --exclude=.git/*
