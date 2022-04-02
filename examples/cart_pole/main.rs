use std::process::exit;

use evolutionary_self_replication::{Environment, Habitat};
use favannat::{matrix::feedforward::evaluator::MatrixFeedforwardEvaluator, network::Evaluator};
use gym_rs::{ActionType, CartPoleEnv, GifRender, GymEnv};
use set_genome::Parameters;

fn main() {
    let mut habitat = Habitat::new(
        100,
        create_environment,
        Parameters::new("examples/cart_pole/config.toml").expect("Could not parse config.toml"),
    );

    loop {
        habitat.step();
        habitat.reproduce();
    }
}

fn create_environment() -> Box<dyn Environment> {
    let mut env = CartPoleEnv::default();
    let state: Vec<f64> = env.reset();
    Box::new(CartPoleEnvironment(env, state, 0))
}

struct CartPoleEnvironment(CartPoleEnv, Vec<f64>, usize);

impl Environment for CartPoleEnvironment {
    fn step(
        &mut self,
        evaluator: &MatrixFeedforwardEvaluator,
    ) -> evolutionary_self_replication::Status {
        let mut input = self.1.clone();
        input.push(1.0);
        let output = evaluator.evaluate(input);

        let action: ActionType = if output[0] < 0.0 {
            ActionType::Discrete(0)
        } else {
            ActionType::Discrete(1)
        };

        let (state, _reward, done, _info) = self.0.step(action);

        self.1 = state;
        self.2 += 1;

        if self.2 > 1000 {
            println!("evaluator over 1000 steps: {:?}", evaluator);
            render_champion(evaluator);
            exit(0);
        }

        if done {
            println!("organism lived for {} steps", self.2);
        }

        (!done).into()
    }
}

fn render_champion(evaluator: &MatrixFeedforwardEvaluator) {
    println!("rendering champion...");

    let mut env = CartPoleEnv::default();

    let mut render = GifRender::new(540, 540, "examples/cart_pole/champion.gif", 20).unwrap();

    let mut state: Vec<f64> = env.reset();

    let mut end: bool = false;
    let mut steps: usize = 0;
    while !end {
        if steps > 1000 {
            break;
        }
        let mut input = state;
        input.push(1.0);
        let output = evaluator.evaluate(input);

        let action: ActionType = if output[0] < 0.0 {
            ActionType::Discrete(0)
        } else {
            ActionType::Discrete(1)
        };
        let (s, _reward, done, _info) = env.step(action);
        end = done;
        state = s;
        steps += 1;

        env.render(&mut render);
    }
}
