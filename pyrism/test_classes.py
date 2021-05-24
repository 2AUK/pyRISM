import Closures
import Potentials
import IntegralEquations
import Solvers
import Core

new_clos = Closures.Closure("KH").get_closure()
new_pot, new_mix_rule = Potentials.Potential("LJ_AB").get_potential()
new_IE = IntegralEquations.IntegralEquation("XRISM").get_IE()
new_solver = Solvers.Solver("Ng").get_solver()
new_data = Core.RISM_Obj(300, 1, 167101, 3, 3, 1024, 20.48, 10)
new_site = Core.Site(
    "O",
    [78.15, 3.16572, -0.8476],
    [0.00000000e00, 0.00000000e00, 0.00000000e00],
    0.033314,
)
new_species = Core.Species("Oxygen")
print(new_clos)
print(new_pot)
print(new_mix_rule)
print(new_IE)
print(new_solver)
print(new_data)
print(new_site)
print(new_species)
new_species.add_site(new_site)
print(new_species)
