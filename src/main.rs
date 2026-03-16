mod env;
mod model;

use dfdx::prelude::*;
use env::OrbitEnv;
use model::OrbitAgent;


fn main() {
    let dev = Cpu::default();
    let mut env = OrbitEnv::new(None, None, Some(250), Some(0.05));
    let mut agent = OrbitAgent::new(&dev);

    let num_episodes = 5000;

    for episode in 0..num_episodes {
        let mut state = env.reset(None, None);
        let mut total_reward = 0.0;

        for _step in 0..250 {
            let (action, action_idx, _prob) = agent.select_action(state, &dev);
            let result = env.step(action);

            agent.update(
                &dev,
                state, 
                action_idx, 
                result.reward, 
                result.next_state, 
                result.is_done, 
                result.is_truncated
            );

            state = result.next_state;
            total_reward += result.reward;

            if result.is_done || result.is_truncated {
                break;
            }
        }

        if episode % 100 == 0 {
            println!("Episode {}: Total Reward = {:.2}", episode, total_reward);
        }
    }

    // agent.actor.save("actor.npz").expect("Failed to save");
}
