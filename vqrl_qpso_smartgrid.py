import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# ==========================================
# 1. SMART GRID ENVIRONMENT (IEEE 118-BUS)
# ==========================================
class SmartGridEnv:
    def __init__(self):
        # Load IEEE 118-bus system
        self.net = nw.IEEE118()
        # Add Renewable Integration at Bus 10
        pp.create_sgen(self.net, 10, p_mw=50, q_mvar=10, name="Wind Farm")
        
    def get_state(self):
        # State: Line loadings and bus voltages
        pp.runpp(self.net)
        loading = self.net.res_line.loading_percent.values / 100.0 # Normalized
        voltages = self.net.res_bus.vm_pu.values
        # Return a subset of critical features (first 12 for 12 qubits)
        return np.concatenate([loading[:6], voltages[:6]])

    def apply_action(self, gen_adjustments):
        # Adjust generator outputs based on QPSO/VQRL decision
        for i, adj in enumerate(gen_adjustments[:54]):
            self.net.gen.at[i, 'p_mw'] += adj
        pp.runpp(self.net)
        
    def get_reward(self):
        # Reward = Negative of (Congestion + Voltage Deviation)
        pp.runpp(self.net)
        congestion = np.sum(np.maximum(0, self.net.res_line.loading_percent - 100))
        v_dev = np.sum(np.abs(self.net.res_bus.vm_pu - 1.0))
        return -(0.7 * congestion + 0.3 * v_dev)

# ==========================================
# 2. VQRL AGENT (Quantum Circuit)
# ==========================================
n_qubits = 12
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def vqc_policy(weights, state):
    # 1. Angle Encoding of Grid State
    for i in range(n_qubits):
        qml.RY(state[i] * np.pi, wires=i)
    
    # 2. Entangling Layers (Ladder Topology)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    
    # 3. Variational Layers
    for i in range(n_qubits):
        qml.RZ(weights[i], wires=i)
    
    # 4. Measurement (Expectation Values)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# ==========================================
# 3. QUANTUM-BEHAVED PSO (QPSO)
# ==========================================
class QPSO:
    def __init__(self, n_particles, n_dimensions, beta):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.beta = beta # Contraction-expansion coefficient
        self.particles = np.random.uniform(-1, 1, (n_particles, n_dimensions))
        self.pbest = self.particles.copy()
        self.gbest = self.particles[0].copy()
        self.gbest_fit = float('inf')

    def update(self, env):
        mbest = np.mean(self.pbest, axis=0)
        
        for i in range(self.n_particles):
            # Quantum Position Update (Delta Potential Well)
            u = np.random.random(self.n_dimensions)
            phi = np.random.random(self.n_dimensions)
            
            # Local Attractor
            p = phi * self.pbest[i] + (1 - phi) * self.gbest
            
            # Update Position
            sign = np.where(np.random.random(self.n_dimensions) > 0.5, 1, -1)
            self.particles[i] = p + sign * self.beta * np.abs(mbest - self.particles[i]) * np.log(1/u)
            
            # Evaluate Fitness (Smart Grid Response)
            env.apply_action(self.particles[i])
            fit = -env.get_reward()
            
            if fit < self.gbest_fit:
                self.gbest = self.particles[i].copy()
                self.gbest_fit = fit
        return self.gbest

# ==========================================
# 4. MAIN HYBRID TRAINING LOOP
# ==========================================
def main():
    env = SmartGridEnv()
    weights = pnp.random.random(n_qubits, requires_grad=True)
    
    print("Starting VQRL-QPSO Congestion Management...")
    
    for epoch in range(50):
        # 1. Observe Grid
        state = env.get_state()
        
        # 2. Quantum Agent Decision (VQC Output)
        quantum_outputs = vqc_policy(weights, state)
        
        # Use quantum output to tune QPSO Beta (Adaptive control)
        adaptive_beta = 0.5 + 0.5 * np.mean(quantum_outputs)
        
        # 3. Optimization via QPSO
        optimizer = QPSO(n_particles=20, n_dimensions=54, beta=adaptive_beta)
        best_rescheduling = optimizer.update(env)
        
        # 4. Verify Congestion Relief
        env.apply_action(best_rescheduling)
        final_reward = env.get_reward()
        
        print(f"Epoch {epoch} | Beta: {adaptive_beta:.4f} | Reward: {final_reward:.4f}")
        
        # 5. Policy Update (Simplification of Parameter Shift rule)
        # In full implementation, use opt = qml.GradientDescentOptimizer()
        weights = weights + 0.01 * final_reward # Heuristic update for demo

if __name__ == "__main__":
    main()
