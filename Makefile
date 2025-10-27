COPY_FILES = $(patsubst src/%,public/%,$(wildcard src/*) $(wildcard src/*/*))

all: $(COPY_FILES)

public/%: src/%
	echo $@
	cp -r -f -v $< $@

deploy:
	rsync -a --info=name src/ public/ --exclude=.git/*
