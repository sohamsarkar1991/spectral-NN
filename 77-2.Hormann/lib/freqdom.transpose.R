#' Transpose the object at each frequency or lag
#'  
#' @title Transpose pointwise timedom or freqdom object
#' @param X freqdom or timedom object
#' @return object of the same type as XI
#' @export
freqdom.transpose = function(X){
  lags = freqdom.lags(X)
  for (i in 1:length(lags))
    X$operators[i,,] = t(X$operators[i,,])
  X
}

#' @export
t.freqdom = freqdom.transpose

#' @export
t.timedom = freqdom.transpose
