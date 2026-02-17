# cthreadingpi

High-performance C-backed threading replacement for Python 3.14t

**Version:** 0.1.0  
**Author:** Sarenian <ozlohu99@gmail.com>  
**License:** unlicense  
**Python Version:** >=3.14

## Installation

Install from source:

```bash
pip install .
```

or install from pypi:

```bash
pip install cthreadingpi
```

## Recommended Usage

```python
from cthreading import auto_threaded


# You can also use the decorator @auto_threaded
# but i prefer the function call style
# like so:


# @auto_threaded
def main():
    # This function is the entrypoint of the program
    # As long as every single line of code in the project 
    # is executed from this function (even from other functions
    # that are called from functions this function calls) the 
    # program will be threaded automatically.
    print("Hello World")

if __name__ == "__main__":
    auto_threaded(main)()
```
