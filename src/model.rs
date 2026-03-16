use dfdx::prelude::*;
use rand::distr::{Distribution, WeightedIndex}; 
use rand::prelude::*;
use crate::env::{Action, State};


type Device = Cpu;  // well, I do not have a GPU, but if you do...

pub type ActorNetwork = (  // s -> 5 logits
    Linear<4, 64>, 
    ReLU, 
    Linear<64, 64>, 
    ReLU, 
    Linear<64, 5>
);

pub type CriticNetwork = (  // s -> v(s) scalar
    Linear<4, 64>, 
    ReLU, 
    Linear<64, 64>, 
    ReLU, 
    Linear<64, 1>
);

pub struct OrbitAgent {
    pub actor: ActorNetwork,
    pub critic: CriticNetwork,
    pub actor_optimizer: Adam<ActorNetwork, f32, Device>,
    pub critic_optimizer: Adam<CriticNetwork, f32, Device>,
    pub gamma: f32,  // 0.99
    pub beta: f32,  // 0.01
}

impl OrbitAgent {
    pub fn new(device: &Device) -> Self {
        let mut actor = device.build_module::<ActorNetwork, f32>();
        let mut critic = device.build_module::<CriticNetwork, f32>();

        actor.reset_params();
        critic.reset_params();

        // Learning rates alpha_theta = 1e-4, alpha_w = 1e-3
        let actor_optimizer = Adam::new(&actor, AdamConfig { lr: 1e-4, ..Default::default() });
        let critic_optimizer = Adam::new(&critic, AdamConfig { lr: 1e-3, ..Default::default() });

        Self {
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            gamma: 0.99, //
            beta: 0.01,  //
        }
    }

    pub fn select_action(&self, state: State, device: &Device) -> (Action, usize, f32) {
        let state_tensor = device.tensor(state);
        
        let logits = self.actor.forward(state_tensor);
        let probs = logits.softmax();
        let probs_vec = probs.as_vec(); 

        let mut rng = thread_rng();
        let dist = WeightedIndex::new(probs_vec.clone()).unwrap();  // Could have used softmax here, but this is for exploration
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

    pub fn update(&mut self, device: &Device, state: State, action_idx: usize, reward: f32, next_state: State, is_done: bool, is_truncated: bool) {
        let s_t = device.tensor(state);
        let s_next = device.tensor(next_state);

        let v_t = self.critic.forward(s_t.clone());
        let v_next = self.critic.forward(s_next);
        
        // bootstrap if is_truncated
        let target_v = if is_done {
            reward
        } else {
            reward + self.gamma * v_next.array()[0]
        };

        let delta = target_v - v_t.array()[0];

        // L(w) = 0.5 * delta^2
        let v_t = self.critic.forward(s_t.trace(device.tensor_stack())); // Start tracking gradients
        let critic_loss = 0.5 * (v_t - target_v).square();
        let gradients = critic_loss.backward();
        self.critic_optimizer.update(&mut self.critic, &gradients).expect("Critic update failed");

        // L(theta) = -ln(pi(a|s)) * delta - beta * entropy
        let logits = self.actor.forward(s_t.trace(device.tensor_stack()));
        let probs = logits.softmax();
        let log_probs = logits.log_softmax();
        
        let action_log_prob = log_probs.slice((action_idx..action_idx + 1)).sum();
        let entropy = (probs.clone() * log_probs).sum() * -1.0;

        // Policy gradient loss: -log_prob * delta - beta * entropy
        // Note: delta is treated as a constant scalar here (stop_grad)
        let actor_loss = (action_log_prob * (-delta)) - (entropy * self.beta);
        
        let gradients = actor_loss.backward();
        self.actor_optimizer.update(&mut self.actor, &gradients).expect("Actor update failed");
    }
}
