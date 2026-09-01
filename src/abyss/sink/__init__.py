"""Where a finished frame goes.

Three modules rather than one, split along the line that matters to a reader:
`base` is the protocol and the shared size check, `file` writes to disk and is
safe anywhere, and `window` needs a display and so cannot run on g4 or in the
test suite. That is the whole reason for the package - the headless-unsafe code
is one named file instead of a class buried among two that are not.

Import from the module that defines the symbol rather than from here.
"""
