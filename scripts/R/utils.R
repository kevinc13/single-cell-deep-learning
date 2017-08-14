catln <- function(..., sep="") {
    cat(..., "\n", sep=sep)
}

print_divider <- function(char="=", len=50) {
    catln(paste(rep(char, len), collapse=""))
}