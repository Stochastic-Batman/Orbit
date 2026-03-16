# Orbit

[![Rust](https://img.shields.io/badge/Rust-1.94.0-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)

Continuous Navigation via Actor-Critic Methods using Neural Function Approximation

Orbit implements an Advantage Actor-Critic (A2C) agent designed to solve a navigation task in a continuous two-dimensional coordinate space. Instead of using discrete grid states, the agent processes floating-point vectors to determine its movement. This requires the integration of neural networks to approximate the policy and value functions, moving beyond the tabular methods typically used in introductory reinforcement learning.

The primary inspiration for this project is [Mathematical Foundations of Reinforcement Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning) by Shiyu Zhao. It is the best technical book I have read to this point and I highly recommend it (do NOT skip the grey boxes!). While the book utilizes a discrete grid world to explain the fundamental mathematics, Orbit moves into neural function approximation. I am applying the concepts from the final three chapters: Value Function Methods, Policy Gradient Methods, and Actor-Critic Methods.

The implementation uses the `dfdx` crate for neural network operations. This provides the necessary automatic differentiation for backpropagation while allowing me to manually write the reinforcement learning logic, including the temporal difference error and policy gradient updates.

## Project Structure

```text
orbit/
├── Cargo.toml          # Project dependencies and metadata
├── src/
│   ├── main.rs         # Entry point: handles the training loop
│   ├── env.rs          # The MDP: continuous navigation world logic
│   ├── model.rs        # Neural network architectures using dfdx
│   ├── agent.rs        # RL logic: A2C implementation and gradient updates
│   └── utils.rs        # Math helpers and data logging
└── README.md
```

## Environment Setup

These instructions assume you have the Rust toolchain installed. I developed this project using `rustc`/`cargo` version 1.94.0.

### 1. Initialize the project

Create the directory structure using cargo:

```bash
cargo new orbit
cd orbit
```

### 2. Add dependencies

Add the following to your `Cargo.toml` file. This configuration includes `dfdx` for neural networks, `rand` for stochastic exploration, and `rayon` for parallelizing environment interactions.

```toml
[package]
name = "orbit"
version = "0.1.0"
edition = "2024"

[dependencies]
dfdx = "0.13.0" 
rand = "0.10.0"
rayon = "1.11.0"
```

### 3. Build the project

To compile the dependencies and the project, run:

```bash
cargo build
```
