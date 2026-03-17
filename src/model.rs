use dfdx::prelude::*;


pub type Device = Cpu;  // well, I do not have a GPU, but if you do...

pub type ActorNetwork = (  // s -> 5 logits
    Linear<4, 64>, ReLU, 
    Linear<64, 64>, ReLU, 
    Linear<64, 5>
);

pub type CriticNetwork = (  // s -> v(s) scalar
    Linear<4, 64>, ReLU, 
    Linear<64, 64>, ReLU, 
    Linear<64, 1>
);


pub type BuiltActor = <ActorNetwork as BuildOnDevice<Device, f32>>::Built;
pub type BuiltCritic = <CriticNetwork as BuildOnDevice<Device, f32>>::Built;
