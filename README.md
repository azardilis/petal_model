The models works with rules that are generalisations of the regular reactions usually found in discrete simulations of chemical reactions.

In the petal model we have Cells that have the following attributes:
   - a cell id, `cid` (integer)
   - a neighbour id, `neigh` (integer), that denotes the cid of the right neighbour of the cell
   - a conc, `conc` (float), that denotes a concentration that in this case encodes the fate of the cell (1 if purple/bottom, 0 if white/top)
   - an area, `area` (float), that in this 1d case it's just the length of the cell

A growth rule in this case is:
```
Cell(cid=i, conc=c, area=a) --> Cell(area=a+g(c))
```
where the growth amount depends on the fate.

This corresponds to multiple concrete reactions:
```
Cell(cid=1, conc=1, ...) --> Cell(...)
Cell(cid=2, conc=1, ...) --> Cell(...)
Cell(cid=3, conc=1, ...) --> Cell(...)
...

```
and so on. In Python rules have type: Set[Cell] -> List[Rxn[Cell]].

The algorithm then is the normal Gillespie algorithm with an extra step for creating the concrete reactions at every step.


To run the petal model for 100 steps or until some condition:
```
import core as c
import petal_model as p

rules = p.mkRules(2, 1)
ss = c.simulate(rules, p.init, nsteps=100) #run for 100 steps
ss = m.simulate_until(rules, init,
                      lambda s: sum([c.area for c in s.state]) > 100) #run until total size is less than 100
```
where in the `mkRules` function, the first argument is the ratio of the growth rates of the bottom vs the top fate and the second
argument is the ratio of the division rates of the top vs the bottom fates.

