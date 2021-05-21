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

print(new_clos)
print(new_pot)
print(new_mix_rule)
print(new_IE)
print(new_solver)
print(new_data)
