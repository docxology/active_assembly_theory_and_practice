## GNNSection
ProbabilisticModel
## GNNVersionAndFlags
GNN v1.1
## ModelName
finite_sensor
## StateSpaceBlock
likelihood[2,2, type=float]
cost[1, type=float]
s[2, type=float]
o[2, type=float]
a[1, type=float]
## Connections
s>o
likelihood>o
a>o
cost>a
## InitialParameterization
likelihood=[[0.8,0.2],[0.2,0.8]]
cost=[1]
## ModelAnnotation
{"adapter":"finite_sensor","parameters":{"likelihood":{"rank":2,"stochastic_axis":0},"cost":{"rank":1,"nonnegative":true}},"nodes":{"s":{"from":"likelihood","axes":[1]},"o":{"from":"likelihood","axes":[0]},"a":{"shape":[1]}},"interpretation":"p(outcome|state,sensor)=likelihood; finite-horizon policy trades expected entropy reduction against explicit sensor cost."}
