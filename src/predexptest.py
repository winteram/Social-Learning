# Test the predictive accuracy and error for different rep sizes
from collections import defaultdict
from numpy import *
from scipy import *
from scipy import optimize

truelambda = [10.0,3.0]
fullobs = floor(truelambda[0]*random.exponential(truelambda[1], 100))

nsamples = 1000

# take in repertoire, return best fitting exponential parameters and error estiamtes
def fitExp( rep ):

    # count frequency of each payoff
    exphist = defaultdict(int)
    for x in rep: exphist[x] += 1

    fitfunc = lambda p,x: p[0] * exp(p[1] * x)
    errfunc = lambda p,x,y: fitfunc(p,x) - y
    lambinit = [9,2]
    expkeys = array(exphist.keys())
    expval = array(exphist.values())
    lamb, success = optimize.leastsq(errfunc, lambinit[:], args=(expkeys,expval))
    fiterr = errfunc(lamb,expkeys,expval)

    return (lamb, sqrt(mean(fiterr**2)))

for i in range(97):
    final = {"pmax":0,"maxdiff":[],"ampdiff":[],"lambdiff":[],"fiterr":[]}
    for n in range(nsamples):
        random.shuffle(fullobs)
        rep = fullobs[0:i+3]
        hasmax = max(rep) == max(fullobs) and 1 or 0
        maxdiff = max(fullobs) - max(rep)
        (lamb, fiterr) = fitExp( rep )
        ampdiff = abs(truelambda[0] - lamb[0])
        lambdiff = abs(truelambda[1] - lamb[1])
        final["pmax"] = hasmax + final["pmax"]
        final["maxdiff"].append(maxdiff)
        final["ampdiff"].append(ampdiff)
        final["lambdiff"].append(lambdiff)
        final["fiterr"].append(fiterr)
    print str(i+3) + "," + str(final["pmax"]) + "," + str(mean(final["maxdiff"])) + "," + str(std(final["maxdiff"])) + "," + str(mean(final["ampdiff"])) + "," + str(std(final["ampdiff"])) + "," + str(mean(final["lambdiff"])) + "," + str(std(final["lambdiff"])) + "," + str(mean(final["fiterr"])) + "," + str(std(final["fiterr"]))
