use std::collections::HashSet;

use std::hash::Hash;

use favannat::{
    matrix::feedforward::{
        evaluator::MatrixFeedforwardEvaluator, fabricator::MatrixFeedForwardFabricator,
    },
    network::{Evaluator, Fabricator},
};
use rand::Rng;
use set_genome::{Genome, GenomeContext, Parameters};

pub struct Habitat {
    capacity: usize,
    organisms: HashSet<Organism>,
    create_environment: fn() -> Box<dyn Environment>,
    genome_context: GenomeContext,
}

impl Habitat {
    pub fn new(
        capacity: usize,
        create_environment: fn() -> Box<dyn Environment>,
        parameters: Parameters,
    ) -> Self {
        Self {
            capacity,
            create_environment,
            genome_context: GenomeContext::new(parameters),
            organisms: HashSet::new(),
        }
    }

    pub fn step(&mut self) {
        if self.organisms.is_empty() {
            for _ in 0..(self.capacity as f64 * 0.1).ceil() as usize {
                let mut genome = self.genome_context.uninitialized_genome();
                genome.init_with_context(&mut self.genome_context);
                self.organisms
                    .insert(Organism::new(genome, (self.create_environment)()));
            }
            self.organisms.insert(Organism::new(
                self.genome_context.initialized_genome(),
                (self.create_environment)(),
            ));
        }

        self.organisms = self
            .organisms
            .drain()
            .filter_map(|mut organism| (organism.step() == Status::Alive).then(|| organism))
            .collect();
    }

    pub fn reproduce(&mut self) {
        let mut new_organisms = Vec::new();
        for organism in &self.organisms {
            if self.organisms.len() < self.capacity && rand::thread_rng().gen::<f64>() < 0.1 {
                // println!("REPRODUCTION");
                let mut genome = organism.genome.clone();
                genome.mutate_with_context(&mut self.genome_context);
                new_organisms.push(Organism::new(genome, (self.create_environment)()))
            }
        }
    }
}

pub struct Organism {
    genome: Genome,
    phenotype: MatrixFeedforwardEvaluator,
    environment: Box<dyn Environment>,
}

impl Organism {
    fn new(genome: Genome, environment: Box<dyn Environment>) -> Self {
        Self {
            phenotype: MatrixFeedForwardFabricator::fabricate(&genome).expect("fabricatio failed"),
            genome,
            environment,
        }
    }

    fn step(&mut self) -> Status {
        self.environment.step(&self.phenotype)
    }
}

impl Hash for Organism {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.genome.hash(state);
    }
}

impl PartialEq for Organism {
    fn eq(&self, other: &Self) -> bool {
        self.genome == other.genome
    }
}

impl Eq for Organism {}

#[derive(Debug, Eq, PartialEq)]
pub enum Status {
    Dead,
    Alive,
}

impl From<bool> for Status {
    fn from(status: bool) -> Self {
        if status {
            Status::Alive
        } else {
            Status::Dead
        }
    }
}

pub trait Environment {
    fn step(&mut self, evaluator: &MatrixFeedforwardEvaluator) -> Status;
}
