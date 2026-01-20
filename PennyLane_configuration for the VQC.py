import pennylane as qml

n_qubits = 12
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def grid_vqc_circuit(weights, state):
    # Encoding Layer
    for i in range(n_qubits):
        qml.RY(state[i] * np.pi, wires=i)
        
    # Variational Layers (Repeated for depth D)
    for layer_weights in weights:
        # 1. Entanglement
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        
        # 2. Rotations
        for i in range(n_qubits):
            qml.RY(layer_weights[i], wires=i)
            
    # Readout
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]