## GNNSection
ProbabilisticModel
## GNNVersionAndFlags
GNN v1.1
## ModelName
identity_phase
## StateSpaceBlock
A[2,2, type=float]
D[2, type=float]
s[2, type=float]
o[2, type=float]
## Connections
D>s
s>o
A>o
## InitialParameterization
A=[[0.8,0.2],[0.2,0.8]]
D=[0.5,0.5]
## ModelAnnotation
{"adapter":"identity_phase","parameters":{"A":{"rank":2,"stochastic_axis":0},"D":{"rank":1,"stochastic_axis":0}},"nodes":{"s":{"from":"D","axes":[0]},"o":{"from":"A","axes":[0]}},"interpretation":"One unchanged latent hypothesis; q_t(s) proportional to A[o_t,s] q_(t-1)(s)."}
