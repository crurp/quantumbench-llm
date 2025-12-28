#!/usr/bin/env python3
"""
generate_quantumbench_data.py

Generate comprehensive QuantumBench dataset with 100+ quantum computing questions.
Covers various topics: qubits, gates, algorithms, entanglement, measurement, etc.
"""

import json
from pathlib import Path


def generate_quantumbench_dataset():
    """
    Generate comprehensive QuantumBench dataset.
    
    Returns:
        List of dictionaries with instruction-response pairs
    """
    questions_and_answers = [
        # Basic Concepts
        ("What is a qubit?", "A qubit is the basic unit of quantum information, analogous to a classical bit but capable of existing in superposition states where it can be both 0 and 1 simultaneously."),
        ("What is quantum superposition?", "Quantum superposition is a fundamental principle where a quantum system can exist in multiple states simultaneously until it is measured, at which point it collapses to a single state."),
        ("What is quantum entanglement?", "Quantum entanglement is a phenomenon where particles become correlated such that the state of one particle instantaneously affects the state of another, regardless of distance."),
        ("What is quantum measurement?", "Quantum measurement is the process by which a quantum system's state collapses from a superposition to a definite classical state, with probabilities determined by the quantum state."),
        ("What is decoherence?", "Decoherence is the process by which quantum systems lose their quantum properties and become classical due to interaction with their environment."),
        
        # Quantum Gates
        ("What is a quantum gate?", "A quantum gate is an operation that manipulates the state of qubits, analogous to classical logic gates but operating on quantum superpositions and entanglement."),
        ("What does the Hadamard gate do?", "The Hadamard gate creates superposition by transforming |0⟩ to (|0⟩ + |1⟩)/√2 and |1⟩ to (|0⟩ - |1⟩)/√2."),
        ("What is a CNOT gate?", "The CNOT (Controlled-NOT) gate is a two-qubit gate that flips the target qubit if and only if the control qubit is |1⟩, creating entanglement."),
        ("What does the Pauli-X gate do?", "The Pauli-X gate is the quantum equivalent of a classical NOT gate, flipping |0⟩ to |1⟩ and |1⟩ to |0⟩."),
        ("What does the Pauli-Y gate do?", "The Pauli-Y gate performs a rotation around the Y-axis of the Bloch sphere, equivalent to XZ up to a global phase."),
        ("What does the Pauli-Z gate do?", "The Pauli-Z gate performs a phase flip, leaving |0⟩ unchanged but flipping |1⟩ to -|1⟩."),
        ("What is a Toffoli gate?", "The Toffoli gate is a three-qubit universal reversible gate that flips the third qubit if both control qubits are |1⟩."),
        ("What is a phase gate?", "A phase gate applies a phase shift to a qubit, rotating it around the Z-axis of the Bloch sphere by a specified angle."),
        
        # Quantum Algorithms
        ("What is Grover's algorithm?", "Grover's algorithm is a quantum search algorithm that can find an item in an unsorted database of N items with O(√N) queries, providing a quadratic speedup over classical algorithms."),
        ("What is Shor's algorithm?", "Shor's algorithm is a quantum algorithm for integer factorization that can factor large numbers exponentially faster than classical algorithms, threatening current RSA cryptography."),
        ("What is the quantum Fourier transform?", "The quantum Fourier transform is a linear transformation on quantum bits that is the quantum analogue of the discrete Fourier transform, used in many quantum algorithms including Shor's algorithm."),
        ("What is amplitude amplification?", "Amplitude amplification is a technique in quantum computing that increases the probability of measuring a desired state, used in Grover's algorithm."),
        ("What is quantum phase estimation?", "Quantum phase estimation is an algorithm that estimates the eigenvalue of a unitary operator, a key subroutine in Shor's algorithm."),
        ("What is the variational quantum eigensolver?", "The variational quantum eigensolver (VQE) is a hybrid quantum-classical algorithm that finds ground state energies of molecules by optimizing a parameterized quantum circuit."),
        ("What is QAOA?", "The Quantum Approximate Optimization Algorithm (QAOA) is a hybrid algorithm that uses quantum circuits to solve combinatorial optimization problems."),
        
        # Quantum States and Circuits
        ("What is a Bell state?", "A Bell state is one of four maximally entangled two-qubit states that serve as a basis for quantum teleportation and superdense coding."),
        ("What is the Bloch sphere?", "The Bloch sphere is a geometric representation of the pure state space of a two-level quantum system, where the north and south poles represent |0⟩ and |1⟩."),
        ("What is a quantum circuit?", "A quantum circuit is a sequence of quantum gates applied to qubits, representing a quantum algorithm or computation."),
        ("What is quantum parallelism?", "Quantum parallelism refers to the ability of quantum computers to process multiple computational paths simultaneously due to superposition."),
        ("What is quantum interference?", "Quantum interference is the phenomenon where probability amplitudes can constructively or destructively interfere, enabling quantum algorithms to amplify correct answers."),
        
        # Quantum Error Correction
        ("What is quantum error correction?", "Quantum error correction uses redundancy and quantum codes to protect quantum information from errors caused by decoherence and noise."),
        ("What is a logical qubit?", "A logical qubit is an error-corrected qubit encoded across multiple physical qubits, providing protection against errors."),
        ("What is surface code?", "Surface code is a topological quantum error correction code that uses a two-dimensional lattice of physical qubits to encode logical qubits with high error thresholds."),
        ("What is stabilizer formalism?", "Stabilizer formalism is a framework for describing quantum error correcting codes and Clifford circuits using group theory."),
        
        # Quantum Communication
        ("What is quantum teleportation?", "Quantum teleportation is a protocol for transferring quantum states between distant locations using entanglement and classical communication, without physically moving the qubit."),
        ("What is superdense coding?", "Superdense coding is a quantum communication protocol that allows two classical bits of information to be transmitted using a single qubit and shared entanglement."),
        ("What is quantum key distribution?", "Quantum key distribution (QKD) uses quantum mechanics to securely exchange cryptographic keys, with security guaranteed by the laws of physics."),
        ("What is BB84 protocol?", "BB84 is a quantum key distribution protocol developed by Bennett and Brassard in 1984, using photon polarization states to create secure keys."),
        
        # Quantum Hardware
        ("What is a quantum processor?", "A quantum processor is a device that manipulates qubits using quantum gates to perform quantum computations."),
        ("What are superconducting qubits?", "Superconducting qubits are quantum bits implemented using superconducting circuits that operate at cryogenic temperatures."),
        ("What are trapped ion qubits?", "Trapped ion qubits use individual ions held in electromagnetic traps as qubits, with high coherence times and gate fidelities."),
        ("What are photonic qubits?", "Photonic qubits use photons as the quantum information carrier, advantageous for quantum communication and certain algorithms."),
        ("What is NISQ?", "NISQ (Noisy Intermediate-Scale Quantum) refers to current quantum computers with 50-100 qubits that have significant noise but can perform useful tasks."),
        
        # Quantum Measurement and Observables
        ("What is a quantum observable?", "A quantum observable is a measurable property of a quantum system, represented by a Hermitian operator with eigenvalues corresponding to possible measurement outcomes."),
        ("What is the expectation value?", "The expectation value is the average result of measuring an observable on a quantum state, calculated as ⟨ψ|A|ψ⟩ for operator A and state |ψ⟩."),
        ("What is projective measurement?", "Projective measurement is a measurement that projects a quantum state onto one of the eigenstates of the measured observable."),
        ("What is weak measurement?", "Weak measurement is a type of quantum measurement that extracts partial information about a quantum system with minimal disturbance."),
        
        # Quantum Information Theory
        ("What is quantum information?", "Quantum information is information encoded in quantum states, allowing for new types of computation and communication impossible with classical information."),
        ("What is von Neumann entropy?", "Von Neumann entropy is a measure of quantum information and entanglement, defined as S(ρ) = -Tr(ρ log ρ) for density matrix ρ."),
        ("What is quantum mutual information?", "Quantum mutual information measures the total correlations between two quantum systems, including both classical and quantum correlations."),
        ("What is the no-cloning theorem?", "The no-cloning theorem states that it is impossible to create an identical copy of an arbitrary unknown quantum state."),
        ("What is the no-communication theorem?", "The no-communication theorem states that quantum entanglement cannot be used to transmit information faster than light."),
        
        # Quantum Algorithms (Continued)
        ("What is the Deutsch-Jozsa algorithm?", "The Deutsch-Jozsa algorithm is a quantum algorithm that determines whether a function is constant or balanced with a single query, while classical algorithms require up to 2^(n-1)+1 queries."),
        ("What is Simon's algorithm?", "Simon's algorithm is a quantum algorithm that finds the period of a function with exponential speedup over classical algorithms."),
        ("What is quantum walk?", "Quantum walk is the quantum analogue of classical random walks, showing different spreading behavior and used in various quantum algorithms."),
        ("What is adiabatic quantum computation?", "Adiabatic quantum computation is a model where a quantum system evolves slowly from a known ground state to find the ground state of a problem Hamiltonian."),
        
        # Quantum Complexity
        ("What is BQP?", "BQP (Bounded-error Quantum Polynomial time) is the class of decision problems solvable by a quantum computer in polynomial time with bounded error probability."),
        ("What is the quantum speedup?", "Quantum speedup refers to the advantage quantum algorithms provide over classical algorithms, such as exponential or polynomial speedups."),
        ("What is quantum supremacy?", "Quantum supremacy (quantum advantage) is the demonstration that a quantum computer can solve a problem that classical computers cannot solve in a reasonable time."),
        
        # Quantum Chemistry and Simulation
        ("What is quantum simulation?", "Quantum simulation uses quantum computers to simulate quantum systems, such as molecules or materials, that are difficult to simulate classically."),
        ("What is the quantum chemistry problem?", "The quantum chemistry problem involves finding electronic structures and properties of molecules by solving the Schrödinger equation, which is exponentially hard classically."),
        ("What is the Hubbard model?", "The Hubbard model is a simplified model for interacting electrons in a lattice, important in condensed matter physics and a target for quantum simulation."),
        
        # Quantum Machine Learning
        ("What is quantum machine learning?", "Quantum machine learning combines quantum computing with machine learning, potentially offering speedups for certain learning tasks."),
        ("What is a quantum neural network?", "A quantum neural network uses quantum circuits with trainable parameters to perform machine learning tasks."),
        ("What is quantum kernel methods?", "Quantum kernel methods use quantum feature maps to create kernels for classical machine learning algorithms, potentially providing advantages for certain data."),
        
        # Advanced Topics
        ("What is topological quantum computing?", "Topological quantum computing uses anyons and topological phases of matter for quantum computation, offering inherent error protection."),
        ("What are anyons?", "Anyons are quasiparticles that exist in two-dimensional systems and have fractional statistics, different from bosons and fermions."),
        ("What is measurement-based quantum computing?", "Measurement-based quantum computing performs computations by measuring entangled quantum states in a specific pattern, rather than applying gates."),
        ("What is the cluster state?", "The cluster state is a highly entangled state of qubits arranged in a lattice, used as a resource for measurement-based quantum computation."),
        
        # Quantum Cryptography
        ("What is quantum cryptography?", "Quantum cryptography uses quantum mechanical properties to provide secure communication, with security based on physical principles rather than computational assumptions."),
        ("What is quantum money?", "Quantum money is a cryptographic protocol that uses quantum states to create currency that cannot be counterfeited due to the no-cloning theorem."),
        ("What is blind quantum computation?", "Blind quantum computation allows a client to delegate quantum computation to a server while keeping the input, output, and algorithm private."),
        
        # Practical Applications
        ("What are quantum sensors?", "Quantum sensors use quantum effects to achieve high precision measurements beyond classical limits, useful in navigation, imaging, and fundamental physics."),
        ("What is quantum metrology?", "Quantum metrology uses quantum resources like entanglement to improve measurement precision beyond classical limits."),
        ("What are the applications of quantum computing?", "Quantum computing applications include cryptography breaking (Shor's algorithm), optimization (QAOA), machine learning, drug discovery, financial modeling, and quantum simulation."),
        
        # Quantum Gates (Advanced)
        ("What is the S gate?", "The S gate is a phase gate that applies a π/2 phase shift, transforming |1⟩ to i|1⟩."),
        ("What is the T gate?", "The T gate is a phase gate that applies a π/4 phase shift, transforming |1⟩ to e^(iπ/4)|1⟩."),
        ("What is the SWAP gate?", "The SWAP gate exchanges the states of two qubits, swapping |01⟩ with |10⟩."),
        ("What is the Fredkin gate?", "The Fredkin gate is a three-qubit gate that swaps the second and third qubits if the first qubit is |1⟩."),
        
        # Quantum Error Models
        ("What is depolarizing noise?", "Depolarizing noise is a quantum noise model where a qubit has a probability of being replaced by a maximally mixed state."),
        ("What is amplitude damping?", "Amplitude damping is a noise model representing energy dissipation, where |1⟩ decays to |0⟩ with a certain probability."),
        ("What is phase damping?", "Phase damping is a noise model representing loss of phase coherence without energy loss."),
        
        # Quantum Algorithms (More)
        ("What is HHL algorithm?", "The HHL (Harrow-Hassidim-Lloyd) algorithm solves linear systems of equations exponentially faster than classical algorithms, under certain conditions."),
        ("What is the quantum counting algorithm?", "The quantum counting algorithm determines the number of solutions to a search problem, generalizing Grover's algorithm."),
        ("What is quantum amplitude estimation?", "Quantum amplitude estimation is an algorithm that estimates the amplitude of a target state in a quantum superposition."),
        
        # Quantum Communication (More)
        ("What is entanglement swapping?", "Entanglement swapping is a protocol that creates entanglement between two particles that have never directly interacted, using measurement."),
        ("What is quantum repeater?", "Quantum repeaters extend the range of quantum communication by creating entanglement over long distances through entanglement swapping."),
        ("What is quantum networking?", "Quantum networking connects quantum processors and devices over quantum channels, enabling distributed quantum computing."),
        
        # Quantum Information Processing
        ("What is quantum compression?", "Quantum compression reduces the number of qubits needed to represent quantum information while preserving all quantum properties."),
        ("What is quantum data compression?", "Quantum data compression uses quantum information theory to efficiently store and transmit quantum states."),
        ("What is the Holevo bound?", "The Holevo bound limits the amount of classical information that can be extracted from a quantum state, fundamental to quantum information theory."),
        
        # Additional Practical Questions
        ("How does a quantum computer differ from a classical computer?", "Quantum computers use quantum mechanics principles like superposition and entanglement to process information, allowing certain problems to be solved exponentially faster than classical computers."),
        ("What makes quantum algorithms faster?", "Quantum algorithms leverage superposition, entanglement, and quantum interference to explore multiple computational paths simultaneously and amplify correct answers."),
        ("What are the main challenges in building quantum computers?", "Main challenges include maintaining quantum coherence, error correction, scaling to many qubits, reducing noise, and creating reliable quantum gates."),
        ("What is quantum volume?", "Quantum volume is a metric that measures the computational power of a quantum computer, considering both the number of qubits and their quality (coherence, gate fidelities, etc.)."),
        ("What is gate fidelity?", "Gate fidelity measures how accurately a quantum gate performs its intended operation, typically expressed as a percentage or error rate."),
        ("What is coherence time?", "Coherence time is the duration for which a qubit maintains its quantum state before decoherence destroys the quantum information."),
        ("What is quantum annealing?", "Quantum annealing is a quantum computing approach that uses quantum fluctuations to find optimal solutions to optimization problems."),
        ("What is a quantum advantage?", "Quantum advantage (quantum supremacy) is demonstrated when a quantum computer solves a problem faster than any classical computer could."),
        ("What is a qubit register?", "A qubit register is a collection of qubits that are manipulated together to perform quantum computations, analogous to classical computer registers."),
        ("What is quantum circuit depth?", "Quantum circuit depth is the number of sequential gate layers in a quantum circuit, a measure of circuit complexity and execution time."),
        ("What is quantum circuit width?", "Quantum circuit width is the number of qubits used in a quantum circuit, representing the quantum parallelism available."),
        ("What is the quantum threshold theorem?", "The quantum threshold theorem states that quantum error correction can work if error rates are below a certain threshold, enabling fault-tolerant quantum computation."),
        ("What is fault-tolerant quantum computing?", "Fault-tolerant quantum computing uses quantum error correction to perform reliable computations even with noisy quantum hardware."),
        ("What is a quantum oracle?", "A quantum oracle is a black box that performs a specific function on quantum states, used as a subroutine in quantum algorithms like Grover's and Deutsch-Jozsa."),
        ("What is quantum complexity theory?", "Quantum complexity theory studies the computational complexity of problems when solved using quantum computers, defining complexity classes like BQP."),
        ("What is quantum state tomography?", "Quantum state tomography is the process of reconstructing a quantum state's density matrix through measurements, essential for characterizing quantum systems."),
        ("What is quantum process tomography?", "Quantum process tomography characterizes quantum operations by determining how they transform quantum states, important for validating quantum gates."),
    ]
    
    # Format as instruction-response pairs
    dataset = []
    for question, answer in questions_and_answers:
        dataset.append({
            "instruction": f"Answer the following quantum computing question:\n\n{question}",
            "response": answer
        })
    
    return dataset


def main():
    """Generate and save QuantumBench dataset."""
    print("Generating QuantumBench dataset...")
    
    dataset = generate_quantumbench_dataset()
    
    print(f"Generated {len(dataset)} examples")
    
    # Save to JSONL
    output_path = Path('data/quantumbench.jsonl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✓ Dataset saved to {output_path}")
    print(f"  Total examples: {len(dataset)}")


if __name__ == "__main__":
    main()

        ("What is a qubit register?", "A qubit register is a collection of qubits that are manipulated together to perform quantum computations, analogous to classical computer registers."),
        ("What is quantum circuit depth?", "Quantum circuit depth is the number of sequential gate layers in a quantum circuit, a measure of circuit complexity and execution time."),
        ("What is quantum circuit width?", "Quantum circuit width is the number of qubits used in a quantum circuit, representing the quantum parallelism available."),
        ("What is the quantum threshold theorem?", "The quantum threshold theorem states that quantum error correction can work if error rates are below a certain threshold, enabling fault-tolerant quantum computation."),
        ("What is fault-tolerant quantum computing?", "Fault-tolerant quantum computing uses quantum error correction to perform reliable computations even with noisy quantum hardware."),
        ("What is a quantum oracle?", "A quantum oracle is a black box that performs a specific function on quantum states, used as a subroutine in quantum algorithms like Grover's and Deutsch-Jozsa."),
        ("What is quantum complexity theory?", "Quantum complexity theory studies the computational complexity of problems when solved using quantum computers, defining complexity classes like BQP."),
        ("What is quantum state tomography?", "Quantum state tomography is the process of reconstructing a quantum state's density matrix through measurements, essential for characterizing quantum systems."),
        ("What is quantum process tomography?", "Quantum process tomography characterizes quantum operations by determining how they transform quantum states, important for validating quantum gates."),
