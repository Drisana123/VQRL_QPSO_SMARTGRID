import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pennylane as qml

# --- 1. Smart Grid Environment ---
class GridEnv:
    def __init__(self):
        self.net = nw.case118() # IEEE 118-Bus
        pp.create_sgen(self.net, 10, p_mw=50, name="Wind Farm") # RES Integration

    def get_state(self):
        pp.runpp(self.net)
        # Normalize top 12 critical features for 12 qubits
        return np.concatenate([self.net.res_line.loading_percent.values[:6]/100, 
                               self.net.res_bus.vm_pu.values[:6]])

# --- 2. Quantum RL Agent (12 Qubits) ---
dev = qml.device("default.qubit", wires=12)

@qml.qnode(dev)
def quantum_policy(weights, state):
    for i in range(12): qml.RY(state[i] * np.pi, wires=i) # Encoding
    for i in range(11): qml.CNOT(wires=[i, i+1]) # Entanglement
    return [qml.expval(qml.PauliZ(i)) for i in range(2)] # Actions: Beta, Center

# --- 3. Quantum-behaved PSO ---
class QPSO:
    def __init__(self, beta):
        self.beta = beta
        self.particles = np.random.uniform(-1, 1, (20, 54)) # 54 Gens in IEEE 118
        self.gbest = self.particles[0]

    def solve(self, env):
        mbest = np.mean(self.particles, axis=0)
        for i in range(20):
            u, phi = np.random.random(54), np.random.random(54)
            p = phi * self.particles[i] + (1-phi) * self.gbest
            # Delta Potential Update
            self.particles[i] = p + np.sign(np.random.random(54)-0.5) * \
                                self.beta * np.abs(mbest - self.particles[i]) * np.log(1/u)
        return self.gbest

# --- 4. Main Execution ---
env = GridEnv()
weights = np.random.random(12)
q_out = quantum_policy(weights, env.get_state())
optimizer = QPSO(beta=0.5 + 0.5*q_out[0])
best_plan = optimizer.solve(env)

print(f"Optimal Rescheduling Plan Found. Quantum Beta: {0.5 + 0.5*q_out[0]:.4f}")