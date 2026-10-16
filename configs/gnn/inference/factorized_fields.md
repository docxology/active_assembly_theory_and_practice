## GNNSection
ProbabilisticModel
## GNNVersionAndFlags
GNN v1.1
## ModelName
factorized_fields
## StateSpaceBlock
likelihood[1,2,2, type=float]
B[2,2, type=float]
D[2, type=float]
divisors[1, type=float]
group_divisor[1, type=float]
s[2, type=float]
o[1,2, type=float]
## Connections
D>s
B>s
s>o
likelihood>o
divisors>o
group_divisor>o
## InitialParameterization
likelihood=[[[0.8,0.2],[0.2,0.8]]]
B=[[0.8,0.2],[0.2,0.8]]
D=[0.5,0.5]
divisors=[1]
group_divisor=[1]
## ModelAnnotation
{"adapter":"factorized_fields","parameters":{"likelihood":{"rank":3,"stochastic_axis":1},"B":{"rank":2,"stochastic_axis":0},"D":{"rank":1,"stochastic_axis":0},"divisors":{"rank":1,"positive":true},"group_divisor":{"rank":1,"positive":true}},"nodes":{"s":{"from":"D","axes":[0]},"o":{"from":"likelihood","axes":[0,1]}},"interpretation":"Composite log score sums field log likelihoods with unit divisors. Missing and unseen have separate support."}
