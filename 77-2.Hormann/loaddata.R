library(fda)

# Import PM10 data
pm<-read.csv(file="data/pm10.txt",header = TRUE)
pm.graz<-sqrt(pm[,4])

# N = number of grid points = 48 (half-hourly observations)
# n = number of functional observations = length(pm[,4])/48

N=48
n=length(pm.graz)/N

# Transform the Graz data to matrix (48x182)

pm.discrete=matrix(pm.graz,nrow=N,ncol=n) 
pm.discrete=pm.discrete[,-(90:96)]

# Remove mean and weekend effects

seven=seq(1:25)*7
six=seq(1:25)*7-1
five=seq(1:25)*7-2
four=seq(1:25)*7-3
three=seq(1:25)*7-4
two=seq(1:25)*7-5
one=seq(1:25)*7-6

# Transformation to functional data

args=seq(0,1,length=48)

nrows = 48
n = 175
nbasis = 15
basis = create.fourier.basis(rangeval=c(0, 1), nbasis=nbasis)

Xorg=Data2fd(args,pm.discrete,basis)

#center
m1=rowSums(pm.discrete[,one])/25
m2=rowSums(pm.discrete[,two])/25
m3=rowSums(pm.discrete[,three])/25
m4=rowSums(pm.discrete[,four])/25
m5=rowSums(pm.discrete[,five])/25
m6=rowSums(pm.discrete[,six])/25
m7=rowSums(pm.discrete[,seven])/25

pm.discrete[,one]<-pm.discrete[,one]-m1
pm.discrete[,two]<-pm.discrete[,two]-m2
pm.discrete[,three]<-pm.discrete[,three]-m3
pm.discrete[,four]<-pm.discrete[,four]-m4
pm.discrete[,five]<-pm.discrete[,five]-m5
pm.discrete[,six]<-pm.discrete[,six]-m6
pm.discrete[,seven]<-pm.discrete[,seven]-m7

# Transform data into functional time serie, evaluate on the grid
X = center.fd(Data2fd(args,pm.discrete,basis))
