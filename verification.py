import math_util as mut

"""verification calcul fractions"""
fractions = [0.2, 0.3, 0.4]
deb_turb = mut.frac_mdot_turbine(fractions, 2)
deb_sout = 0
for i in range(0, len(fractions)):
    deb_sout += mut.frac_mdot_sout(fractions, i)
    print('deb', deb_sout)
if deb_sout + deb_turb == 1:
    print('valide')
else:
    print('non valide')