# Test the predictive accuracy and error for different rep sizes
from collections import defaultdict
from numpy import *
from scipy import *
from scipy import optimize


nsamples = 1000
possible_fcns = ["expon","gamma","pois","uniform","powerlaw","lognorm"]


# take in repertoire, return best fitting exponential parameters and error estiamtes

def fitPwr( rep ):

    # count frequency of each payoff
    exphist = defaultdict(int)
    for x in rep: exphist[x] += 1

    # transform to linear
    expkeys = log(array(exphist.keys())+1)
    expval = log(array(exphist.values()))

    fitfunc = lambda p,x: p[0] + p[1] * x
    errfunc = lambda p,x,y: y - fitfunc(p,x)
    lambinit = [log(10),log(2)]
    lamb, success = optimize.leastsq(errfunc, lambinit[:], args=(expkeys,expval))
    fiterr = errfunc(lamb,expkeys,expval)

    return (exp(lamb), sqrt(mean(fiterr**2)))





for amp in range(20,50,5):
    for truelamb in [x * 0.5 for x in range(1, 10)]:
        truelambda = [amp,truelamb]
        final = {"amp":[], "lamb":[], "fiterr":[]}
        for n in range(nsamples):
            fullobs = floor(truelambda[0]*random.random(100)**truelambda[1])
            (lamb, fiterr) = fitPwr( fullobs )
            final["amp"].append(lamb[0])
            final["lamb"].append(lamb[1])
            final["fiterr"].append(fiterr)
        print str(amp) + "," + str(truelamb) + "," + str(mean(final["amp"])) + "," + str(std(final["amp"])) + "," + str(mean(final["lamb"])) + "," + str(std(final["lamb"])) + "," + str(mean(final["fiterr"])) + "," + str(std(final["fiterr"]))
