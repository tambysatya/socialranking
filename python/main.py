
from MIP.kp import generateUniformKP, generateWeaklyCorrelatedKP, generateStronglyCorrelatedKP

#f= [1,2]
#A = [[2,3],[2,5]]
#bs = [3,3]
#
#p = Problem (f,A,bs)
#
#print (p.solve())
#
#print (p.pb.variables.get_upper_bounds())
#print (p.solve_coalition({1}))
#print (p.pb.variables.get_upper_bounds())

kp = generateUniformKP(10,20)
print(kp.solve())
print(kp.solve_coalition({2,5,8,9}))

