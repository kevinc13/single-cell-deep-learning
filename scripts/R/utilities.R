# ------------------------------------------------------------------------------
# Helper functions
# Author: Kevin Chen
# ------------------------------------------------------------------------------

catln <- function(..., sep="") {
    # Similar to the 'cat' function, but adds a newline at the end
    # 
    # Args: 
    #   ...: Arbitrary number of objects to print
    #   sep: Separator between objects

    cat(..., "\n", sep=sep)
}

print_divider <- function(char="=", len=80) {
    # Prints a divider
    # 
    # Args: 
    #   char: Character to repeat
    #   len: Length of divider
    catln(paste(rep(char, len), collapse=""))
}