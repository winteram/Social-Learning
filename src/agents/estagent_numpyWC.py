from moves import * 
# bring in standard names for moves
# this means INNOVATE, OBSERVE, EXPLOIT and REFINE can be used instead of -1,0,1,2 
# and that AGE, TOTAL_PAY, TIMES_COPIED and N_OFFSPRING can be used to index into exploiterData

import numpy as np


def move(roundsAlive,rep,historyRounds,historyMoves,historyActs, historyPayoffs, historyDemes, currentDeme, canChooseModel, canPlayRefine, multipleDemes):
    epsilon = 0.001

    def Exponential_cdf(x, lambda1):
        expon_cdf = 1-np.exp(-lambda1*x)
        return expon_cdf

    def Poisson_pmf(x, lambda1):
        Poiss_pmf = (lambda1**x) * np.exp(-lambda1) / np.factorial(x)
        return Poiss_pmf

    def Poisson_cdf(x, lambda1):
        Poiss_cdf = 0
        for i in range(0, x+1) :
            Poiss_cdfTmp = Poisson_pmf(i, lambda1)
            Poiss_cdf = Poiss_cdf + Poiss_cdfTmp
        return Poiss_cdf

    def estimate_payoff(CurrMax, E_lifeleft, DistName, params) :
        Delta = 0.5
        ErrorFlag = 0
        EstPayoff_total = 0

        CurrPayoff = float(CurrMax) / float((E_lifeleft-1))


        if DistName == 'Expon' :
            assert len(params)==2
            (loc, lamb) = params
            i = 1
            EstPos_temp = Exponential_cdf(CurrMax + i + Delta, lamb) - Exponential_cdf(CurrMax + i - Delta, lamb)
            while EstPos_temp > epsilon:
                EstPos_temp = Exponential_cdf(CurrMax + i + Delta, lamb) - Exponential_cdf(CurrMax + i - Delta, lamb)
                # for continuous distribution, P[X=x] is calculated as Cdf[x+0.5] - Cdf[x-0.5]. 
                EstPayoff_temp = EstPos_temp * i
                EstPayoff_total = EstPayoff_total + EstPayoff_temp
                i = i+1

        elif DistName == 'Poiss' :
            assert len(params)==2
            (loc,mu) = params
            i = 1
            EstPos_temp = Poisson_pmf(CurrMax + i, mu)
            while EstPos_temp > epsilon:
                EstPos_temp = Poisson_pmf(CurrMax + i, mu)
                EstPayoff_temp= EstPos_temp * i
                EstPayoff_total = EstPayoff_total + EstPayoff_temp
                i = i+1

        else :
            ErrorFlag = 1 # set a error flag

        #print "Current Payoff is : " , CurrPayoff
        #print "Estimated Payoff is : " , EstPayoff_total

        if (EstPayoff_total > CurrPayoff + 1) and (ErrorFlag == 0):
            #print "Estimated Payoff is larger, take Observation: "
            Decision_result = 0
        elif (EstPayoff_total > CurrPayoff) and  (ErrorFlag == 0) :
            #print "Estimated Payoff is less than +1, take Refine: "
            Decision_result = 2
        elif (EstPayoff_total <= CurrPayoff) and  (ErrorFlag == 0) :
            #print "Estimated Payoff is smaller, take actions: "
            Decision_result = 1
        else :
            print "An error in estimation: ", DistName
            Decision_result = 0

        return Decision_result


    def est_fit(values):
        counts = np.asarray(np.bincount(values), dtype= float)
        x = np.asarray(range(0,len(counts)))
        idx = [i for i,j in enumerate(counts) if j == 0]
        xvals = [i for j, i in enumerate(x) if j not in idx]
        y = [i for j, i in enumerate(counts) if j not in idx]
        logy = np.log(y) 
        A = np.vstack([xvals, np.ones(len(xvals))]).T
        params = np.linalg.lstsq(A, logy)[0]
        return(params[1],-1.5*params[0])



    REGROUND = 2 # number of rounds to observe before making decisions when moved demes
    ACTTHRESH = 10 # number of actions to know before deciding to act
    EXPECTED_LIFETIME = 50
    
    if roundsAlive==0:
        return (OBSERVE,)
    if roundsAlive==1 and len(rep.items())==0:
        return (INNOVATE,)
    else:
        # check if demes are enabled
        if multipleDemes:
            steps_back = min(len(historyDemes),REGROUND)
            if historyDemes[-steps_back:].count(currentDeme) < steps_back and len(rep.items())<ACTTHRESH:
                return (OBSERVE,)

        if len(rep.items()) < ACTTHRESH or np.std(rep.values())==0:
            sacrifice = np.random.random()
            if len(historyRounds)>10 and len(rep.items())>0:
                if sacrifice < 0.05:
                    act = sorted(rep, key=rep.get, reverse=True)[0]
                    return(EXPLOIT,act)
                elif sacrifice < 0.1:
                    return (INNOVATE,)
                else:
                    return (OBSERVE,)
            else:
                return (OBSERVE,)
        else:
            params = est_fit(rep.values())
            if params[0] < epsilon:
                return (OBSERVE,)
            exp_decision = estimate_payoff(max(rep.values()),max(EXPECTED_LIFETIME-roundsAlive,10),'Expon',params)
            if exp_decision == 1:
                act = sorted(rep, key=rep.get, reverse=True)[0]
                return (EXPLOIT,act)
            elif exp_decision == 2 and canPlayRefine:
                act = sorted(rep, key=rep.get, reverse=True)[0]
                return (REFINE,act)
            else:
                return (OBSERVE,)

def observe_who(exploiterData):
    #'This function MUST return the given list of tuples, exploiterData, sorted by preference for copying.'
    #'Data given for each agent are (index in this list,age,total accrued payoff,number of times copied,number of offpsring)'
    #'All values except index have error applied'
    # older agents have observed more
    # total_payoffs have accessed high value actions
    return sorted(exploiterData,key=lambda x:alpha*x[TOTAL_PAY]+(1-alpha)*(x[AGE]*x[N_OFFSPRING]),reverse=True) # copy most rewarded


