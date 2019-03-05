# etsy-shop-top-terms
Demo project retrieving data from a set of Etsy shops and determining the top terms for each shop


* The program requires Python 3.6 or higher.
* The terms" include single-word and multi-word keyphrases.
* One of the demo shops closed during development. I left it in the list of shops in order to demonstrate proper error handling.
    
For standard use, simply do `./terms.py` with no options.
   
There are various command-line options available; please do `./terms.py -h` for usage.
    Notably, `--debug` will show a lot more information than the standard usage, and `--quiet` will only show results.

Documentation is in docs/build/html -- if you have Sphinx installed you can build it yourself thus:
    `cd docs`
    `make html`
