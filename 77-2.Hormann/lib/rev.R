#' @export
rev.timedom = function(XI){
  XI$lags = rev(XI$lags)
  XI
}

#' @export
rev.freqdom = function(XI){
  XI$freq = rev(XI$freq)
  XI
}
