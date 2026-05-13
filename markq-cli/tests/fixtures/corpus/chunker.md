# Markdown chunker

The chunker splits markdown sources on heading boundaries and never splits
a fenced code block. Tokens are counted via a pluggable tokenizer.

## Targets

Default chunk target is 900 tokens with 15 percent overlap.
