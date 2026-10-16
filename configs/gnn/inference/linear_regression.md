## GNNSection
ProbabilisticModel
## GNNVersionAndFlags
GNN v1.1
## ModelName
linear_regression
## StateSpaceBlock
coefficients[1, type=float]
design[1, type=float]
prediction[1, type=float]
## Connections
design>prediction
coefficients>prediction
## InitialParameterization
coefficients=[0.0]
## ModelAnnotation
{"adapter":"linear_regression","parameters":{"coefficients":{"rank":1}},"nodes":{"design":{"from":"coefficients","axes":[0]},"prediction":{"shape":[1]}},"interpretation":"Least-squares timing calibration: prediction=design dot coefficients. This deterministic calibration mapping does not define a new probability distribution or a general cost law."}
