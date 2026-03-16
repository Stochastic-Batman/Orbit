use rand::RngExt; 


pub type State = [f32; 4];

pub enum Action {
    North, // a_1
    East,  // a_2
    South, // a_3
    West,  // a_4
    Stay,  // a_5
}

pub struct StepResult {
    pub next_state: State,
    pub reward: f32,
    pub is_done: bool,
    pub is_truncated: bool, 
}

pub struct OrbitEnv {
    agent_pos: [f32; 2],
    target_pos: [f32; 2],
    current_step: u32,
    max_steps: u32,
    step_size: f32,
}

impl OrbitEnv {

    pub fn new(agent_pos: Option<[f32; 2]>, target_pos: Option<[f32; 2]>, max_steps: Option<u32>, step_size: Option<f32>) -> Self {
        let mut rng = rand::rng(); 

        Self {
            agent_pos: agent_pos.unwrap_or_else(|| [rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0)]),
            target_pos: target_pos.unwrap_or_else(|| [rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0)]),
            current_step: 0,
            max_steps: max_steps.unwrap_or(250),
            step_size: step_size.unwrap_or(0.05),
        }
    }

    /// Reset the environment for a new episode.
    pub fn reset(&mut self, agent_pos: Option<[f32; 2]>, target_pos: Option<[f32; 2]>) -> State {
        let mut rng = rand::rng(); 
        
        self.agent_pos = agent_pos.unwrap_or_else(|| [rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0)]);
        self.target_pos = target_pos.unwrap_or_else(|| [rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0)]);
        self.current_step = 0;

        self.get_state()
    }

    pub fn get_state(&self) -> State {
        [self.agent_pos[0], self.agent_pos[1], self.target_pos[0], self.target_pos[1]]
    }

    pub fn step(&mut self, action: Action) -> StepResult {
        self.current_step += 1;

        let delta = match action {
            Action::North => [0.0, self.step_size],
            Action::East  => [self.step_size, 0.0],
            Action::South => [0.0, -self.step_size],
            Action::West  => [-self.step_size, 0.0],
            Action::Stay  => [0.0, 0.0],
        };

        let next_pos = [self.agent_pos[0] + delta[0], self.agent_pos[1] + delta[1]];

        let reward = if next_pos[0] < -1.0 || next_pos[0] > 1.0 || next_pos[1] < -1.0 || next_pos[1] > 1.0 
        {
            -2.85  
        } else {
            -Self::calculate_distance(next_pos, self.target_pos)
        };

        self.agent_pos[0] = next_pos[0].clamp(-1.0, 1.0);
        self.agent_pos[1] = next_pos[1].clamp(-1.0, 1.0);

        let dist_to_target = Self::calculate_distance(self.agent_pos, self.target_pos);
        let is_done = dist_to_target < 0.05;
        let is_truncated = self.current_step >= self.max_steps;

        StepResult {
            next_state: self.get_state(),
            reward,
            is_done,
            is_truncated,
        }
    }

    fn calculate_distance(p1: [f32; 2], p2: [f32; 2]) -> f32 {
        ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt()
    }
}
