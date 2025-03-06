#' For given frequency domain operator, compute its eigenvalues and eigendirections
#' such that for close frequencies eigendirection matrices are close to each other
#' ('quasi-continuity').
#'
#' @title Eigendevompose a frequency domain operator at each frequency
#' @param S frequency domain operator
#' @return Rotated matrix \code{M}
#' @export
freqdom.eigen = function(S){
  if (!is.freqdom(S))
    stop("S must be a freqdom object")
  
  op = S$operators[1,,]
  if (dim(op)[1] != dim(op)[2])
    stop("S$operators[,,theta] must be a square matrix")
  
  thetas = S$freq
	E = list()
	nbasis = dim(S$operators)[2]
  T = length(thetas)
  
  E$freq = S$freq
	E$vectors = array(0,c(T,nbasis,nbasis))
	E$values = array(0,c(T,nbasis))
	Prev = diag(nbasis)
  
	for (theta in 1:T)
	{
		Eg = close.eigen(S$operators[theta,,],Prev)
		Prev = Eg$vectors

		E$vectors[theta,,] = Eg$vectors
		E$values[theta,] = Eg$values
	}
	E
}

# Function \code{close.eigen} finds the eigenvalues of \code{M} close to\code{Prev}.
close.eigen = function(M,Prev){  
  Eg = eigen(M)
	V = Re(Eg$vectors)
	W = Re(Prev)
	nbasis = dim(M)[1]

	for (col in 1:nbasis){		
		if (sum(V[,col] * W[,col]) < 0)
			Eg$vectors[,col] = -Eg$vectors[,col]
	}

	Eg
}
