use dfdx::prelude::*;
use dfdx::optim::Adam;
use rand::distr::weighted::WeightedIndex; 
use rand::prelude::*;
use std::fs;
use std::path::Path;

use crate::env::{Action, State};
use crate::model::{BuiltActor, BuiltCritic, ActorNetwork, CriticNetwork, Device};


pub struct OrbitAgent {
    pub actor: BuiltActor,
    pub critic: BuiltCritic,
    pub actor_optimizer: Adam<BuiltActor, f32, Device>,
    pub critic_optimizer: Adam<BuiltCritic, f32, Device>,
    pub gamma: f32,  // 0.99
    pub beta: f32,  // 0.01
}

impl OrbitAgent {
    pub fn new(device: &Device) -> Self {
        let actor = device.build_module::<ActorNetwork, f32>();
        let critic = device.build_module::<CriticNetwork, f32>();

        let actor_optimizer = Adam::new(&actor, AdamConfig { lr: 1e-4, ..Default::default() });
        let critic_optimizer = Adam::new(&critic, AdamConfig { lr: 1e-3, ..Default::default() });

        Self {
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            gamma: 0.99,
            beta: 0.01,
        }
    }

    pub fn select_action(&self, state: State, device: &Device) -> (Action, usize, f32) {
        let state_tensor = device.tensor(state);
        let logits = self.actor.forward(state_tensor);
        let probs = logits.softmax();
        let probs_vec = probs.as_vec();

        let mut rng = rand::rng();
        let dist = WeightedIndex::new(probs_vec.clone()).unwrap();
        let action_idx = dist.sample(&mut rng);

        let action = match action_idx {
            0 => Action::North,
            1 => Action::East,
            2 => Action::South,
            3 => Action::West,
            _ => Action::Stay,
        };

        (action, action_idx, probs_vec[action_idx])
    }

    pub fn update(&mut self, device: &Device, state: State, action_idx: usize, reward: f32, next_state: State, is_done: bool, _is_truncated: bool) {
        let s_t = device.tensor(state);
        let s_next = device.tensor(next_state);

        // Forward passes for values (no gradients needed here)
        let v_t_val = self.critic.forward(s_t.clone()).array()[0];
        let v_next_val = self.critic.forward(s_next).array()[0];

        // bootstrap if not a terminal state (is_done)
        let target_v = if is_done {
            reward
        } else {
            reward + self.gamma * v_next_val
        };

        let delta = target_v - v_t_val;

        // L(w) = 0.5 * delta^2
        let grads = self.critic.alloc_grads();
        let v_t = self.critic.forward(s_t.clone().traced(grads));
        let critic_loss = (v_t - target_v).square().sum() * 0.5;
        let gradients = critic_loss.backward();
        self.critic_optimizer.update(&mut self.critic, &gradients).expect("Critic update failed");

        // L(theta) = -ln(pi(a|s)) * delta - beta * entropy
        let grads = self.actor.alloc_grads();
        let logits = self.actor.forward(s_t.traced(grads));
        let (logits_val, tape) = logits.split_tape();
        let log_probs = logits_val.clone().log_softmax();
        let probs = logits_val.softmax();

        let action_log_prob = log_probs.clone().slice((action_idx..action_idx + 1,)).sum();
        let entropy = (probs * log_probs).sum() * -1.0;

        let actor_loss = (action_log_prob * (-delta)) - (entropy * self.beta);
        let gradients = actor_loss.put_tape(tape).backward();

        self.actor_optimizer.update(&mut self.actor, &gradients).expect("Actor update failed");
    }

    pub fn save(&self, episode: usize, base_path: &str) {
        let dir_path = format!("{}/episode_{}", base_path, episode);
        fs::create_dir_all(&dir_path).expect("Could not create save directory");

        self.actor.save(format!("{}/actor.npz", dir_path)).expect("Actor save failed");
        self.critic.save(format!("{}/critic.npz", dir_path)).expect("Critic save failed");
        self.actor_optimizer.save(format!("{}/actor_optim.npz", dir_path)).expect("Actor optim save failed");
        self.critic_optimizer.save(format!("{}/critic_optim.npz", dir_path)).expect("Critic optim save failed");
        
        println!("Successfully saved model state to {}", dir_path);
    }

    pub fn load(&mut self, episode: usize, base_path: &str) {
        let dir_path = format!("{}/episode_{}", base_path, episode);
        
        self.actor.load(format!("{}/actor.npz", dir_path)).expect("Actor load failed");
        self.critic.load(format!("{}/critic.npz", dir_path)).expect("Critic load failed");
        self.actor_optimizer.load(format!("{}/actor_optim.npz", dir_path)).expect("Actor optim load failed");
        self.critic_optimizer.load(format!("{}/critic_optim.npz", dir_path)).expect("Critic optim load failed");

        println!("Resumed training from episode {}", episode);
    }
}
