import matplotlib.pyplot as plt
 
 
plt.figure(figsize=(20, 10), dpi=100)
game = [0,1,2,3,4,5,6,7,8,9,10]
# scores = [0.6488409456047739, 0.6495294927702547, 0.6497590084920817, 0.6511361028230434, 0.6499885242139086, 0.64838191416112, 0.6453982097773697, 0.6431030525591003, 0.639889832453523, 0.635988065182465, 0.6334633922423686]
scores = [0.6488409456047739, 0.6456277254991967, 0.326371356437916, 0.05370667890750516, 0.017902226302501722, 0.018131742024328667, 0.017902226302501722, 0.017672710580674777, 0.017672710580674777, 0.017672710580674777,0.017672710580674777]
plt.ylim(0,0.7)
plt.plot(game, scores)
plt.savefig('tets.png')