#' For a given multivariate stationary time series estimates a covarianve matrix
#' \eqn{C_{XY}^k = Cov(X_k,Y_0)} using the formula
#' \deqn{\hat C_{XY}^k =  \frac{1}{n} \sum_{i=1}^{n-k} X_{k+i} Y_k'. }
#'
#' @title Compute cross covariance with a given lag
#' @param X first process 
#' @param Y second process, if null then autocovariance of X is computed
#' @param lag the lag that we are interested in
#' @return Covariance matrix 
#' @export
#' @examples
#' X = rar(100)
#' Y = rar(100)
#' lagged.cov(X,Y)
lagged.cov = function(X,Y=NULL,lag=0){
  if (is.null(Y))
		Y = X

  if (dim(X)[1] != dim(Y)[1])
    stop("Number of observations must be equal")
  if (!is.matrix(X) || !is.matrix(Y))
    stop("X and Y must be matrices")
  
	n = dim(X)[1]
	h = abs(lag)
	
  if (n - 1 <= h)
	  stop(paste("Too little observations to compute lagged covariance with lag",h))
	
  M = t(X[1:(n-h),]) %*% (Y[1:(n-h)+h,])/(n)
	if (lag < 0){
	  M = t(Y[1:(n-h),]) %*% (X[1:(n-h)+h,])/(n)
	  M = t(M)
	}
	M
}

