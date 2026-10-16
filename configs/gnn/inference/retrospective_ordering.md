## GNNSection
ProbabilisticModel
## GNNVersionAndFlags
GNN v1.1
## ModelName
retrospective_ordering
## StateSpaceBlock
A[2,2, type=float]
preferences[2, type=float]
s[2, type=float]
o[2, type=float]
a[1, type=float]
## Connections
s>o
A>o
o>a
preferences>a
## InitialParameterization
A=[[0.8,0.2],[0.2,0.8]]
preferences=[0.5,0.5]
## ModelAnnotation
{"adapter":"retrospective_ordering","parameters":{"A":{"rank":2,"nonnegative":true},"preferences":{"rank":1,"stochastic_axis":0}},"nodes":{"s":{"from":"A","axes":[1]},"o":{"from":"A","axes":[0]},"a":{"shape":[1]}},"interpretation":"Category-informed ordering diagnostic; observed candidate categories enter scoring. A contains conditional nonnegative weights; prospective objectives additionally require a normalized channel. Preferences here are probabilities, distinct from dimensionless POMDP C utility."}
