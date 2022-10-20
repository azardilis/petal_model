from typing import *
from core import Rxn, Rule, SimState
from dataclasses import dataclass
import numpy as np
import pyvista as pv #type: ignore
import core as m
from functools import partial
from itertools import product
import matplotlib.pyplot as plt #type: ignore
import math


RateF = Callable[[float], float]


@dataclass(frozen=True)
class Cell:
    cid: int
    neigh: int
    conc: float
    area: float

    
# rules
def grate(h: int, c: float) -> float:
    base = 0.05
    if c == 1:
        return base*h
    else:
        return base
    
def divrate(h: int, c: float) -> float:
    base = 0.05
    if c == 1:
        return base
    else:
        return base*h


def grate_(h: int, area: float, c: float) -> float:
    area_max = 5
    const = 0.5
    base = const / (1 + h)

    if c == 1:
        k_growth = base*h
    else:
        k_growth = base

    return k_growth*area*(1 - (area/area_max))

    
def divrate_(h: int, c: float) -> float:
    const = 0.05
    base = const / (1 + h)
    if c == 1:
        return base
    else:
        return base*h


def grate__(gp: float, gw: float, c: float) -> float:
    if c == 1:
        return gp
    else:
        return gw


def divrate__(dp: float, fw: float, c: float) -> float:
    if c == 1:
        return 0.1
    else:
        return 1.0


def growth(rf: RateF, m1: Set[Cell]) -> List[Rxn[Cell]]:
    rxns = []
    for c in m1:
        rxn = Rxn(lhs=set([c]),
                  rhs=set([Cell(c.cid, c.neigh,
                                c.conc, c.area+1)]),
                  rate=rf(c.conc))
        
        rxns.append(rxn)
        
    return rxns


def growth_(rf, m1: Set[Cell]) -> List[Rxn[Cell]]:
    rxns = []
    for c in m1:
        rxn = Rxn(lhs=set([c]),
                  rhs=set([Cell(c.cid, c.neigh,
                                c.conc, c.area+rf(c.area, c.conc))]),
                  rate=1.0)
        
        rxns.append(rxn)
        
    return rxns
        
        
def div(rf: RateF, m1: Set[Cell]) -> List[Rxn[Cell]]:
    rxns = []
    n = len(m1)
    for c in m1:
        rxn = Rxn(lhs=set([c]),
                  rhs=set([Cell(cid=c.cid, neigh=n+1,
                                area=c.area/2, conc=c.conc),
                           Cell(cid=n+1, neigh=c.neigh,
                                area=c.area/2, conc=c.conc)]),
                  rate=rf(c.conc))
        rxns.append(rxn)
    
    return rxns


def mkRules(growth_h, div_h):
    gf = partial(grate_, growth_h)
    df = partial(divrate_, div_h)

    return [partial(growth_, gf), partial(div, df)]


def go_params(ps):
    gp, gw, dp, dw = ps
    def mk_grate_f(gp, gw):
        return partial(grate__, gp, gw)

    def mk_drate_f(dp, dw):
        return partial(divrate__, dp, dw)

    gf = mk_grate_f(gp, gw)
    df = mk_drate_f(dp, dw)
                
    rules = [partial(growth, gf), partial(div, df)]
    ss = m.simulate(rules, init, 500)

    areas = [c.area for c in orderCells(ss[-1].state)]
    xs = np.cumsum(areas)
    plt.plot(xs, areas, 'o')
    plt.show()


def orderCells(state: Set[Cell]) -> List[Cell]:
    state_ = dict()
    for c in state:
        state_[c.cid] = c

    ostate = []
    i = 1
    while i != -1:
        c = state_[i]
        ostate.append(c)
        i = c.neigh
        
    return ostate


def mkCells(ostate: List[Cell]):
    ylen = 5
    lens = np.cumsum([c.area for c in ostate])
    v1 = np.array([0, ylen, 0], dtype=float)
    v2 = np.array([0, 0, 0], dtype=float)
    vs = [(0, v1), (1, v2)]
    fs = []
    for (i, c), l in zip(enumerate(ostate, 1), lens):
        v1, v2 = [i for i, v in vs[-2:]]
        v1_ = np.array([l, ylen, 0], dtype=float)
        v2_ = np.array([l, 0, 0], dtype=float)
        vs.append((v2+1, v1_))
        vs.append((v2+2, v2_))
        fs.append([4, v2, v1, v2+1, v2+2])


    vs_  = [v for i, v in vs]
    mesh = pv.PolyData(vs_, np.hstack(fs))
    mesh.cell_data["conc"] = [c.conc for c in ostate]

    return mesh


def get_mesh(st: Set[Cell]):
    ostate = orderCells(st)
    return mkCells(ostate)


init_st: Set[Cell] = set([Cell(1, 2, 1, 0.01),
                          Cell(2, 3, 1, 0.01),
                          Cell(3, 4, 1, 0.01),
                          Cell(4, 5, 0, 0.01),
                          Cell(5, 6, 0, 0.01),
                          Cell(6, 7, 0, 0.01),
                          Cell(7, 8, 0, 0.01),
                          Cell(8, 9, 0, 0.01),
                          Cell(9, 10, 0, 0.01),
                          Cell(10, -1, 0, 0.01)])


init_st_: Set[Cell] = set([Cell(1, 2, 1, 0.05),
                           Cell(2, 3, 1, 0.05),
                           Cell(3, 4, 0, 0.01),
                           Cell(4, 5, 0, 0.01),
                           Cell(5, 6, 0, 0.01),
                           Cell(6, 7, 0, 0.01),
                           Cell(7, 8, 0, 0.01),
                           Cell(8, 9, 0, 0.01),
                           Cell(9, 10, 0, 0.01),
                           Cell(10, -1, 0, 0.01)])


init: SimState[Cell] = SimState(init_st, 0.0, 0)

init_: SimState[Cell] = SimState(init_st_, 0.0, 0)


def mkInitState(ncells: int, length: float=0.1, ratio: float=0.3):
    cells = list()
    cell_len = length / ncells
    m = math.floor(ncells * ratio)
    for i in range(ncells-1):
        if i < m:
            c = Cell(i, i+1, 1, cell_len)
        else:
            c = Cell(i, i+1, 0, cell_len)

        cells.append(c)

    cells.append(Cell(i+1, -1, 0, cell_len))

    return SimState(set(cells), 0.0, 0)


def get_bound(ss: List[SimState]) -> List[float]:
    bs = []
    for s in ss:
        s_ = orderCells(s.state)
        area1 = sum([c.area for c in s_ if c.conc==1])
        area_total = sum([c.area for c in s_])
        bs.append(area1 / area_total)
     
    return bs


def go_prod() -> Dict[Tuple[float, float], List[List[SimState]]]:
    d = dict()
    growth_hs = [1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5]
    div_hs = [1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5]

    for gh, dh in product(growth_hs, div_hs):
        print(gh, dh)
        rules = mkRules(gh, dh)
        sss = go_n_until(rules, init, 5)
        d[(gh, dh)] = sss

    return d


def go_n_until(rules: List[Rule],
               init: SimState[Cell],
               n: int) -> List[List[SimState]]:
    sss = []
    for i in range(n):
        print(i)
        ss = m.simulate_until(rules, init,
                              lambda s: sum([c.area for c in s.state]) > 100)
        sss.append(ss)

    return sss


def go_n(rules: List[Rule],
         init: SimState[Cell],
         n: int,
         nsteps: int=100) -> List[List[SimState]]:
    sss = []
    for i in range(n):
        print(i)
        ss = m.simulate(rules, init, nsteps)
                              
        sss.append(ss)

    return sss


def to_meshes(ss):
    for i, s in enumerate(ss):
        m = mkCells(orderCells(s.state))
        m.save(f"petal_cells{i}.vtk")
