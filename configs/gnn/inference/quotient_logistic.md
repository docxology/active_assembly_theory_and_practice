## GNNSection
ProbabilisticModel
## GNNVersionAndFlags
GNN v1.1
## ModelName
quotient_logistic
## StateSpaceBlock
weights[1, type=float]
bias[1, type=float]
center[1, type=float]
scale[1, type=float]
x[1, type=float]
y[1, type=float]
## Connections
x>y
weights>y
bias>y
center>y
scale>y
## InitialParameterization
weights=[0]
bias=[0]
center=[0]
scale=[1]
## ModelAnnotation
{"adapter":"quotient_logistic","parameters":{"weights":{"rank":1},"bias":{"rank":1},"center":{"rank":1},"scale":{"rank":1,"positive":true}},"nodes":{"x":{"from":"weights","axes":[0]},"y":{"shape":[1]}},"interpretation":"p(y=1|x)=sigmoid(((x-center)/scale) dot weights + bias). Fit and support use training data only."}
