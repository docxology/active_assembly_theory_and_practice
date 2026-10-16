## GNNSection
ProbabilisticModel
## GNNVersionAndFlags
GNN v1.1
## ModelName
quotient_recurrent
## StateSpaceBlock
wxh[1,1, type=float]
whh[1,1, type=float]
why[1, type=float]
bias[1, type=float]
token[1, type=float]
h_prev[1, type=float]
h[1, type=float]
y[1, type=float]
## Connections
token>h
h_prev>h
wxh>h
whh>h
h>y
why>y
bias>y
## InitialParameterization
wxh=[[0]]
whh=[[0]]
why=[0]
bias=[0]
## ModelAnnotation
{"adapter":"quotient_recurrent","parameters":{"wxh":{"rank":2},"whh":{"rank":2},"why":{"rank":1},"bias":{"rank":1}},"nodes":{"token":{"from":"wxh","axes":[1]},"h_prev":{"from":"why","axes":[0]},"h":{"from":"why","axes":[0]},"y":{"shape":[1]}},"interpretation":"h_t=tanh(wxh[:,token_t]+whh h_(t-1)); p(y=1)=sigmoid(why dot h_T+bias). Training uses explicit bounded back-propagation."}
