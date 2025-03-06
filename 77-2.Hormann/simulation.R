source('library.R')

library(MASS)
library(fda)

psinorm = function(P){sqrt(eigen(P%*%t(P))$values[1])}

RES = c()

comps = c(6,3,2,1)

Lrange = c(60)
components = c(1,6)
norms = c(0.9,0.6,0.3,0.1)
spectrals = c("est")
prange = c(15,101)
decays = c("sqr1","sqr2","exp1")

nsetups = length(Lrange) * length(components) * length(norms) * length(spectrals) * length(prange) * length(decays)

repetitions = 200

# Loop through all the possible setugs
for (setup in 0:(nsetups-1)) {
    ind1 = setup%%length(Lrange)
    setup = (setup-ind1)/length(Lrange)
    ind2 = setup%%length(norms)
    setup = (setup-ind2)/length(norms)
    ind3 = setup%%length(components)
    setup = (setup-ind3)/length(components)
    ind4 = setup%%length(spectrals)
    setup = (setup-ind4)/length(spectrals)
    ind5 = setup%%length(prange)
    setup = (setup-ind5)/length(prange)
    ind6 = setup%%length(decays)
    setup = (setup-ind6)/length(decays)

    for (i in 1:repetitions) {
        # Load setup
        L = Lrange[ind1+1]
        normPsi = norms[ind2+1]
        cmp = components[ind3+1]
        cols = 1:cmp
        spec = spectrals[ind4+1]
        nbasis = prange[ind5+1]
        decay = decays[ind6+1]

        n = 400
        T = 1000

        dec = exp(-(0:(nbasis-1))/10)
        basis = create.fourier.basis(rangeval=c(0, 1), nbasis=nbasis)

        # create an appropriate random Psi operator
        if (decay != "exp1") {
            # 'quadratic' decay
            Decay = matrix(rep(0,nbasis^2),nrow=nbasis)
            for (x in 1:nbasis){
                for (y in 1:nbasis){
                    if (decay=="sqr1")
                        Decay[x,y] = 1/sqrt(x^2 + y^2)
                    if (decay=="sqr2")
                        Decay[x,y] = 1/(x^1.5 + y^1.5)
                }
            }
        }
        else {
            # 'exponential' decay
            dec = exp(-(0:(nbasis-1)))
            decM = rep(dec,nbasis)
            Decay = t(matrix(decM,ncol=nbasis))
            Decay = Decay * t(Decay)
        }
        
        # Generate Psi
        Psi = matrix(rnorm(nbasis*nbasis),ncol=nbasis)
        Psi = Psi * t(Decay)
        Psi = (Psi/psinorm(Psi))*normPsi

        C = dec/sqrt(sum(dec^2))
        noise=function(n){ rnorm(nbasis) * sqrt(C) }

        # generate an AR process
        Xr = rar(n, Psi=Psi,noise=noise)

        #center the data
        X = center.fd(fd(t(Xr), basis=basis ))	# rarh with identity as a Psi
        freq = (-(T):(T))*pi/(T)

        ## Dynamic PCA ##
        XI = dprcomp(t(X$coef),lags=-60:60,freq=freq,weights="Bartlett",q=20) # dpca.coefs(E$vectors,thetas,L)
        Y = filter.process(t(X$coef),XI)
        Y[,-cols] = 0
        Xdpca = t(filter.process( Y, t(rev(XI)) ) )
        Xdpca.fd = fd(Re(Xdpca),basis=X$basis)

        ## Static PCA ##
        PR = prcomp(t(X$coef))
        Y1 = t(X$coef) %*% PR$rotation #PR$x
        Y1[,-cols] = 0
        Xpca = t(Y1 %*% t(PR$rotation))
        Xpca.fd = fd(Xpca,basis=X$basis)

        # save the parameters
        R = c(which(decay == decays), nbasis,normPsi,length(cols))

        # variance explained
        R = c(R,sum((X$coef-Xpca.fd$coef)^2)/sum((X$coef)^2))
        R = c(R,sum((X$coef-Xdpca.fd$coef)^2)/sum((X$coef)^2))

        RES = rbind(RES,R)
    }
}

colnames(RES) = c('Psi','d','normPsi','ncomp','pca','dpca') 


