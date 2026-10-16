## GNNSection
ProbabilisticModel
## GNNVersionAndFlags
GNN v1.1
## ModelName
quotient_markov
## StateSpaceBlock
initial[2,2, type=float]
transition[2,2,2, type=float]
prior[2, type=float]
label[2, type=float]
token_prev[2, type=float]
token[2, type=float]
## Connections
prior>label
label>token_prev
initial>token_prev
token_prev>token
label>token
transition>token
## InitialParameterization
initial=[[0.5,0.5],[0.5,0.5]]
transition=[[[0.5,0.5],[0.5,0.5]],[[0.5,0.5],[0.5,0.5]]]
prior=[0.5,0.5]
## ModelAnnotation
{"adapter":"quotient_markov","parameters":{"initial":{"rank":2,"stochastic_axis":1},"transition":{"rank":3,"stochastic_axis":1},"prior":{"rank":1,"stochastic_axis":0}},"nodes":{"label":{"from":"prior","axes":[0]},"token_prev":{"from":"initial","axes":[1]},"token":{"from":"initial","axes":[1]}},"interpretation":"Class-conditional first-order token-chain likelihood with training-only vocabulary and positive pseudocounts; normalized posterior over pair labels."}
