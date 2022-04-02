use std::process::exit;

use evolutionary_self_replication::{Environment, Habitat};
use favannat::{matrix::feedforward::evaluator::MatrixFeedforwardEvaluator, network::Evaluator};
use gym_rs::{ActionType, GifRender, GymEnv, MountainCarEnv};
use set_genome::Parameters;

// TODO: change to also enable MatrixRecurrentEvaluator

fn main() {
    let mut habitat = Habitat::new(
        100,
        create_environment,
        Parameters::new("examples/mountain_car/config.toml").expect("Could not parse config.toml"),
    );

    loop {
        habitat.step();
        habitat.reproduce();
    }
}

fn create_environment() -> Box<dyn Environment> {
    let mut env = MountainCarEnv::default();
    let state: Vec<f64> = env.reset();
    Box::new(MountainCarEnvironment(env, state, 0))
}

struct MountainCarEnvironment(MountainCarEnv, Vec<f64>, usize);

impl Environment for MountainCarEnvironment {
    fn step(
        &mut self,
        evaluator: &MatrixFeedforwardEvaluator,
    ) -> evolutionary_self_replication::Status {
        let mut input = self.1.clone();
        input.push(1.0);

        let output = evaluator.evaluate(input);

        let action = ActionType::Continuous(output);

        let (state, _reward, mut done, _info) = self.0.step(action);

        if done {
            println!("organism lived for {} steps", self.2);
            println!("evaluator: {:?}", evaluator);
            render_champion(evaluator);
            exit(0);
        }

        if (self.1[0] - state[0]).abs() < 0.0005 {
            done = true;
        }

        self.1 = state;
        self.2 += 1;

        if done {
            println!("organism lived for {} steps", self.2);
        }

        (!done).into()
    }
}

fn render_champion(evaluator: &MatrixFeedforwardEvaluator) {
    println!("rendering champion...");

    let mut env = MountainCarEnv::default();
    env.seed(0);

    let mut render = GifRender::new(540, 540, "examples/mountain_car/champion.gif", 50).unwrap();

    let mut state: Vec<f64> = env.reset();

    let mut end: bool = false;
    let mut steps: usize = 0;
    while !end {
        if steps > 300 {
            break;
        }
        let mut input = state.clone();
        input.push(1.0);

        let output = evaluator.evaluate(input);

        let action = ActionType::Continuous(output);
        let (s, _reward, done, _info) = env.step(action);
        end = done;
        state = s;
        steps += 1;

        env.render(&mut render);
    }
}
