mod env;
mod model;
mod agent;

use std::fs;
use dfdx::prelude::*;
use env::OrbitEnv;
use agent::OrbitAgent;


fn parse_pos(s: &str) -> Option<[f32; 2]> {
    let parts: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    if parts.len() == 2 { 
        Some([parts[0], parts[1]]) 
    } else { 
        None 
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut agent_pos: Option<[f32; 2]> = None;
    let mut target_pos: Option<[f32; 2]> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--agent" if i + 1 < args.len() => {
                agent_pos = parse_pos(&args[i + 1]);
                i += 1;
            }
            "--target" if i + 1 < args.len() => {
                target_pos = parse_pos(&args[i + 1]);
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    let dev = Cpu::default();
    let mut env = OrbitEnv::new(agent_pos, target_pos, Some(250), Some(0.05));
    let mut agent = OrbitAgent::new(&dev);

    let mut start_episode: u32 = 0;
    let model_root = "models";

    if let Ok(entries) = fs::read_dir(model_root) {
        let mut episodes: Vec<u32> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                e.file_name()
                    .into_string()
                    .ok()?
                    .strip_prefix("episode_")?
                    .parse::<u32>()
                    .ok()
            })
            .collect();
        episodes.sort_unstable();

        if let Some(&latest) = episodes.last() {
            let load_path = format!("{}/episode_{:04}", model_root, latest);
            println!("Found checkpoint at episode {}. Loading...", latest);

            agent.actor.load(format!("{}/actor.npz", load_path)).expect("Failed to load actor");
            agent.critic.load(format!("{}/critic.npz", load_path)).expect("Failed to load critic");

            start_episode = latest + 1;
        }
    }

    let num_episodes: u32 = 5000;

    for episode in start_episode..num_episodes {
        let mut state = env.reset(None, None);
        let mut total_reward = 0.0f32;

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
                result.is_truncated,
            );

            state = result.next_state;
            total_reward += result.reward;

            if result.is_done || result.is_truncated {
                break;
            }
        }

        if episode % 100 == 0 {
            println!("Episode {}: Total Reward = {:.2}", episode, total_reward);

            let save_dir = format!("{}/episode_{:04}", model_root, episode);
            fs::create_dir_all(&save_dir).expect("Failed to create save directory");

            agent.actor.save(format!("{}/actor.npz", save_dir)).expect("Failed to save actor");
            agent.critic.save(format!("{}/critic.npz", save_dir)).expect("Failed to save critic");

            println!("Checkpoint saved to {}", save_dir);
        }
    }

    let final_dir = format!("{}/final", model_root);
    fs::create_dir_all(&final_dir).ok();
    agent.actor.save(format!("{}/actor.npz", final_dir)).ok();
    agent.critic.save(format!("{}/critic.npz", final_dir)).ok();
}
