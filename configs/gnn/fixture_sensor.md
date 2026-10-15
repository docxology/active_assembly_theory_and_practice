## GNNSection
ActInfPOMDP
## GNNVersionAndFlags
GNN v1.1
## ModelName
fixture_sensor
## StateSpaceBlock
A0[5,2,3, type=float]
A1[3,3, type=float]
B0[2,2,1, type=float]
B1[3,3,3, type=float]
C0[5, type=float]
C1[3, type=float]
D0[2, type=float]
D1[3, type=float]
E[3, type=float]
## Connections
D0>A0
D1>A0
D1>A1
## InitialParameterization
A0=[[[1.0,0.0,0.0],[1.0,0.0,0.0]],[[0.0,1.0,0.0],[0.0,1.0,0.0]],[[0.0,0.0,0.0],[0.0,0.0,1.0]],[[0.0,0.0,1.0],[0.0,0.0,0.0]],[[0.0,0.0,0.0],[0.0,0.0,0.0]]]
A1=[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
B0=[[[1.0],[0.0]],[[0.0],[1.0]]]
B1=[[[1.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.0,0.0]],[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]],[[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]]]
C0=[0.0,0.0,0.0,0.0,0.0]
C1=[0.0,-0.01,-0.1]
D0=[0.5,0.5]
D1=[1.0,0.0,0.0]
E=[0.3333333333333333,0.3333333333333333,0.3333333333333333]
## Metadata
factors: ["hypothesis", "sensor_mode"]
state_labels: [["x86_accumulator", "x86_bitwise"], ["idle", "static", "dynamic"]]
modalities: ["evidence", "cost"]
actions: [["noop"], ["idle", "static", "dynamic"]]
outcomes: [["missing", "load_immediate", "result_3", "result_5", "unseen"], ["idle", "static", "dynamic"]]
A_dependencies: [[0, 1], [1]]
horizon: 1
gamma: 1.0
seed: 0
