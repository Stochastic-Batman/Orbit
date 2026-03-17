use std::fs::OpenOptions;
use std::io::Write;


pub struct Logger {
    filename: String,
}

impl Logger {
    pub fn new(filename: &str) -> Self {
        let mut file = std::fs::File::create(filename).unwrap();
        writeln!(file, "episode,reward").unwrap(); // CSV Header
        Self { filename: filename.to_string() }
    }

    pub fn log(&self, episode: usize, reward: f32) {
        let mut file = OpenOptions::new()
            .append(true)
            .open(&self.filename)
            .unwrap();
        writeln!(file, "{},{}", episode, reward).unwrap();
    }
}

pub fn print_stats(episode: usize, reward: f32, window_reward: f32) {
    println!(
        "Eps {:<5} | Last Reward: {:10.2} | Avg Reward: {:10.2}",
        episode, reward, window_reward
    );
}
