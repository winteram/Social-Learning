from moves import * #bring in standard names for moves
# this means INNOVATE, OBSERVE, EXPLOIT and REFINE can be used instead of -1,0,1,2 
# and that AGE, TOTAL_PAY, TIMES_COPIED and N_OFFSPRING can be used to index into exploiterData

import random, math

def test(inputstring):
    return "New agent received "+inputstring

def move(roundsAlive,repertoire,historyRounds,historyMoves,historyActs, historyPayoffs, historyDemes, currentDeme, canChooseModel, canPlayRefine, multipleDemes):
#'roundsAlive, currentDeme are integers, history* are tuples of integers'
#'repertoire is a dictionary with keys=behavioural acts and values=payoffs'
#'canChooseModel, canPlayRefine, multipleDemes are boolean, indicating whether:'
#'observe_who will be called (i.e. model bias), REFINE is available (i.e. cumulative), and multiple demes (i.e. spatial) respectively'
#'This function MUST return a tuple in the form (MOVE,ACT) if MOVE is EXPLOIT or REFINE, or (MOVE,) if MOVE is INNOVATE or OBSERVE'
    if roundsAlive>1: #if this isn't my first or second round
        # calculate mean payoff from playing exploit
        myMeanPayoff = sum([p for i,p in enumerate(historyPayoffs) if historyMoves[i]==EXPLOIT])/float(historyMoves.count(EXPLOIT)) 
        lastPayoff = historyPayoffs[len(historyMoves)-1-historyMoves[::- 1].index(EXPLOIT)] #get last payoff from exploit
        lastMove = historyMoves[-1] #get last move
        if lastMove==OBSERVE or lastPayoff>=myMeanPayoff: #if lastMove was observe or lastPayoff at least as good as myMeanPayoff
            if random.random()<0.05 and canChooseModel: #if simulation allows refinement
                return (REFINE,max(repertoire, key=repertoire.get)) #then REFINE best known act with probability 1/20
            else:
                return (EXPLOIT,max(repertoire, key=repertoire.get)) #otherwise EXPLOIT best known act
        else:
            return (OBSERVE,) #if payoffs have dropped then OBSERVE
    elif roundsAlive>0: #if this is my second round
        return (EXPLOIT,repertoire.keys()[0]) #then EXPLOIT the act innovated on the first round
    else: #otherwise this must be the first round
        return (INNOVATE,)


def observe_who(exploiterData):
#'This function MUST return the given list of tuples, exploiterData, sorted by preference for copying.'
#'Data given for each agent are (index in this list,age,total accrued payoff,number of times copied,number of offpsring)'
#'All values except index have error applied'
    return sorted(exploiterData,key=lambda x:x[AGE],reverse=True) #copy oldest
