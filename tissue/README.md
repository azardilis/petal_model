Model using framework: https://gitlab.com/slcu/teamHJ/tissue
(installation instructions there)

Model is in `mol_growth_div.model`, the initial state in `line.init`, and solver parameters in `solver.rk5`.
To run:
```
$SIMULATOR mol_growth_div.model line.init solver.rk5
```

This will create  a folder `vtk` with vtk files that can be viewed in something like Paraview.