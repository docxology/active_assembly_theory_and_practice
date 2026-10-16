## GNNSection
ProbabilisticModel
## GNNVersionAndFlags
GNN v1.1
## ModelName
gaussian_event_code
## StateSpaceBlock
mean[1, type=float]
variance[1, type=float]
shape[1, type=float]
rate[1, type=float]
mu[1, type=float]
tau[1, type=float]
y[1, type=float]
## Connections
mean>mu
variance>mu
shape>tau
rate>tau
mu>y
tau>y
## InitialParameterization
mean=[0]
variance=[100]
shape=[1]
rate=[1]
## ModelAnnotation
{"adapter":"gaussian_event_code","parameters":{"mean":{"rank":1},"variance":{"rank":1,"positive":true},"shape":{"rank":1,"positive":true},"rate":{"rank":1,"positive":true}},"nodes":{"mu":{"shape":[1]},"tau":{"shape":[1]},"y":{"shape":[1]}},"interpretation":"mu~NormalMeanVariance(mean,variance); tau~GammaShapeRate(shape,rate); y~NormalMeanPrecision(mu,tau). Numeric category-code diagnostic."}
