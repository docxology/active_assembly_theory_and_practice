## GNNSection
ProbabilisticModel
## GNNVersionAndFlags
GNN v1.1
## ModelName
platt_calibration
## StateSpaceBlock
slope[1, type=float]
intercept[1, type=float]
logit[1, type=float]
probability[1, type=float]
## Connections
logit>probability
slope>probability
intercept>probability
## InitialParameterization
slope=[1]
intercept=[0]
## ModelAnnotation
{"adapter":"platt_calibration","parameters":{"slope":{"rank":1},"intercept":{"rank":1}},"nodes":{"logit":{"shape":[1]},"probability":{"shape":[1]}},"interpretation":"p_calibrated=sigmoid(slope*logit(p_raw)+intercept), fit on development predictions only."}
