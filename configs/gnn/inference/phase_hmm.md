## GNNSection
ProbabilisticModel
## GNNVersionAndFlags
GNN v1.1
## ModelName
phase_hmm
## StateSpaceBlock
A[2,2, type=float]
B[2,2, type=float]
D[2, type=float]
s_prev[2, type=float]
s[2, type=float]
o[2, type=float]
## Connections
D>s_prev
s_prev>s
B>s
s>o
A>o
## InitialParameterization
A=[[0.8,0.2],[0.2,0.8]]
B=[[0.8,0.2],[0.2,0.8]]
D=[0.5,0.5]
## ModelAnnotation
{"adapter":"phase_hmm","parameters":{"A":{"rank":2,"stochastic_axis":0},"B":{"rank":2,"stochastic_axis":0},"D":{"rank":1,"stochastic_axis":0}},"nodes":{"s_prev":{"from":"D","axes":[0]},"s":{"from":"D","axes":[0]},"o":{"from":"A","axes":[0]}},"interpretation":"p(s_1)=D; p(s_t|s_(t-1))=B; p(o_t|s_t)=A"}
