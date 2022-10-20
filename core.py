import numpy as np
from typing import *
from dataclasses import dataclass
from functools import reduce
from operator import add
import random


T = TypeVar('T', bound=Hashable)


@dataclass(frozen=True)
class Rxn(Generic[T]):
    lhs: Set[T]
    rhs: Set[T]
    rate: float

        
@dataclass(frozen=True)
class SimState(Generic[T]):
    state: Set[T]
    time: float
    steps: int


Rule = Callable[[Set[T]], List[Rxn[T]]]


def madd(m1: Set[T], m2: Set[T]):
    return set.union(m1, m2)


def msub(m1: Set[T], m2: Set[T]):
    return set.difference(m1, m2)


def apply_rxn(rxn: Rxn[T], state: Set[T]) -> Set[T]:
    return madd(msub(state, rxn.lhs), rxn.rhs)


def step(rules, sim_state: SimState[T]) -> SimState[T]:
    rxns = reduce(add, [r(sim_state.state) for r in rules])
    rates = [rxn.rate for rxn in rxns]
    total_prop = np.sum(rates)

    rates_ = [r / total_prop for r in rates]
    a = random.uniform(0, 1)

    rxn = np.random.choice(rxns, p=rates_)
    dt = np.log(1.0 / a) / total_prop

    state_ = apply_rxn(rxn, sim_state.state)
    t = sim_state.time + dt
    steps_ = sim_state.steps + 1

    return SimState(state_, t, steps_)


def simulate(rules: List[Rule],
             state: SimState[T],
             nsteps: int) -> List[SimState[T]]:
    obss = []
    for i in range(nsteps):
        obss.append(state)
        state = step(rules, state)

    return obss


def simulate_until(rules: List[Rule],
                   state: SimState[T],
                   stop_f: Callable[[SimState[T]], bool]) -> List[SimState[T]]:
    obss = []
    while not stop_f(state):
        obss.append(state)
        state = step(rules, state)

    return obss
                   
