from moves import * 
# bring in standard names for moves
# this means INNOVATE, OBSERVE, EXPLOIT and REFINE can be used instead of -1,0,1,2 
# and that AGE, TOTAL_PAY, TIMES_COPIED and N_OFFSPRING can be used to index into exploiterData

import numpy as np
import scipy.stats as ss

def move(roundsAlive,rep,historyRounds,historyMoves,historyActs, historyPayoffs, historyDemes, currentDeme, canChooseModel, canPlayRefine, multipleDemes):
    REGROUND = 2 # number of rounds to observe before making decisions when moved demes
    ACTTHRESH = 10 # number of actions to know before deciding to act
    EXPECTED_LIFETIME = 50
    
    if roundsAlive==0:
        return (OBSERVE,)
    if roundsAlive==1 and len(rep.items())==0:
        return (INNOVATE,)
    else:
        # check if demes are enabled
        #if multipleDemes:
        #    steps_back = min(len(historyDemes),REGROUND)
        #    if historyDemes[-steps_back:].count(currentDeme) < steps_back and len(rep.items())<ACTTHRESH:
        #        return (OBSERVE,)

        if len(rep.items()) < 10 or np.std(rep.values())==0:
            if len(historyRounds)>10 and len(rep.items())>0 and np.random.random()<0.1:
                act = sorted(rep, key=rep.get, reverse=True)[0]
                return(EXPLOIT,act)
            else:
                return (OBSERVE,)
        else:
            params = ss.expon.fit(rep.values())
            exp_decision = estimate_payoff(max(rep.values()),max(EXPECTED_LIFETIME-roundsAlive,10),'Expon',params)
#            params = ss.poisson.fit(rep.values())
#            poi_decision = estimate_payoff(max(rep.values()),max(EXPECTED_LIFETIME-roundsAlive,10),'Poiss',params)
#            params = ss.gamma.fit(rep.values())
#            gma_decision = estimate_payoff(max(rep.values()),max(EXPECTED_LIFETIME-roundsAlive,10),'Gamma',params)
            if exp_decision == 1:
                act = sorted(rep, key=rep.get, reverse=True)[0]
                return(EXPLOIT,act)
            else:
                return(OBSERVE,)

def observe_who(exploiterData):
    #'This function MUST return the given list of tuples, exploiterData, sorted by preference for copying.'
    #'Data given for each agent are (index in this list,age,total accrued payoff,number of times copied,number of offpsring)'
    #'All values except index have error applied'
    # older agents have observed more
    # total_payoffs have accessed high value actions
    return sorted(exploiterData,key=lambda x:x[TOTAL_PAY],reverse=True) # copy most rewarded



def estimate_payoff(CurrMax, E_lifeleft, DistName, params) :
    Delta = 0.5
    ErrorFlag = 0
    EstPayoff_total = 0

    CurrPayoff = float(CurrMax) / float((E_lifeleft-1))


    if DistName == 'Expon' :
        assert len(params)==2
        (loc, lamb) = params
        max_dist = int(round(ss.expon.ppf(0.9999999999, loc, lamb)))
        
        for i in range(1,max_dist):
            EstPos_temp = ss.expon.cdf(CurrMax + i + Delta, loc, lamb) - ss.expon.cdf(CurrMax + i - Delta, loc, lamb)
            # for continuous distribution, P[X=x] is calculated as Cdf[x+0.5] - Cdf[x-0.5]. 
            EstPayoff_temp = EstPos_temp * i
            EstPayoff_total = EstPayoff_total + EstPayoff_temp

    elif DistName == 'Poiss' :
        assert len(params)==2
        (loc,mu) = params
        max_dist = int(round(ss.poisson.ppf(0.9999999999, loc, mu)))

        for i in range(1,max_dist):
            EstPos_temp = ss.poisson.cdf(CurrMax + i, mu, loc=loc)
            EstPayoff_temp= EstPos_temp * i
            EstPayoff_total = EstPayoff_total + EstPayoff_temp

    elif DistName == 'Gamma' :
        assert len(params)==3
        (a,loc,lamb) = params
        max_dist = int(round(ss.gamma.ppf(0.9999999999, a, loc, lamb)))

        for i in range(1,max_dist):
            EstPos_temp = ss.gamma.cdf(CurrMax + i, a, loc=loc, scale=lamb)
            EstPayoff_temp= EstPos_temp * i
            EstPayoff_total = EstPayoff_total + EstPayoff_temp

    else :
        ErrorFlag = 1 # set a error flag
    
    #print "Current Payoff is : " , CurrPayoff
    #print "Estimated Payoff is : " , EstPayoff_total

    if (EstPayoff_total > CurrPayoff) and  (ErrorFlag == 0):
        #print "Estimated Payoff is larger, take Observation: "
        Decision_result = 0
    elif (EstPayoff_total <= CurrPayoff) and  (ErrorFlag == 0) :
        #print "Estimated Payoff is smaller, take actions: "
        Decision_result = 1
    else :
        print "An error in estimation: ", DistName
        Decision_result = 0

    return Decision_result
