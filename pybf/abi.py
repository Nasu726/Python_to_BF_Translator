"""Fixed runtime ABI.

These are compiler/runtime type properties, not command-line configuration.
Changing them is an ABI change and requires rebuilding/retesting the compiler.
"""

INT_BITS = 64
STRING_CAPACITY = 255
LIST_CAPACITY = 64
