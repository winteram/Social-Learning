# Model for Tournament for social learning strategies II contest
# author: Winter Mason (m@winteram.com)

# load required packages
import random
import sys, os
import logging
import pprint
import datetime
from scipy.stats import bernoulli,poisson,norm,expon,uniform
from numpy import mean,std,median,max
from moves import * #bring in standard names for moves

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger=logging.getLogger('SLSTournament')

# Each strategy must have two functions, "move" and "observe"
# "move" takes as input: 
#    canPlayRefine, canChooseModel, currentDeme, roundsAlive, repertoire, history
# "move"  must return: 
#    move, action, currentRep
# "observe_who" takes as input:
#    modelInfo, a list of dictionaries, one for each potential model, with the info:
#       age, total payoff, times observed, # offspring
# "observe_who" must return:
#    models, an ordered list of indices corresponding to modelInfo

# Load agents from agent directory
agent_dir = 'agents/'
agentfiles = os.listdir(agent_dir)
if not agentfiles:
    logger.error("Missing agent directory")
    exit(0)
random.shuffle(agentfiles)
strategies = []
while len(strategies) < 2:
    file = agentfiles.pop()
    if file:
        fname = file.split('.')
        if fname[1] == "py" and fname[0] != "moves" and fname[0] != "__init__":
            #agents = __import__('agents.'+fname[0])
            exec("import agents."+fname[0]+" as "+fname[0])
            strategies.append(fname[0])
    else:
        logger.error("Not enough files")
        exit(0)

# Create output file
outputFname = "../data/SLS_run_"+datetime.datetime.now().strftime("%y%m%d_%H%M")+".csv"
try:
    outputFH = open(outputFname, 'w')
except IOError:
    print 'cannot open', outputFname
    exit(0)



##TODO: Allow importing of parameters with input file
# Initialize parameters in model
canPlayRefine = False # if refine move is available
canChooseModel = False # if observe_who is an option
multipleDemes = False # if spatial extension is enabled
runIn = False # If there is a run-in time for the agents to develop

N = 100 # initial population size
ngen = 1000 # number of generations
nact = 100 # number of possible actions
nObserve = 5 # number of models, between 1--10

lambd = 0.8 # 1 / mean of exponential distribution
pchg = 0.05 # probability of environment changing
pmut = 0.02 # probability of mutation
pmig = 0.03 # probability of migration (if spatial extension is enabled)
pdie = 0.02 # probability of dying

copy_error = 0.01
sigma = 0.2

# structure for a new agent
class newagent:
    Name = "newagent"

    def __init__(self):
        self.strategy = False
        self.rep = {}
        self.lastMove = False
        self.historyRounds =[]
        self.historyMoves =[]
        self.historyActs =[]
        self.historyPayoffs =[] 
        self.historyDemes =[]
        self.born = 0
        self.roundsAlive = 0
        self.nObserved = 0
        self.currentDeme = 0 
        self.pointsEarned = 0
        self.nOffspring =0

    def show(self):
        print "Strategy: " + self.strategy
        print "Repertoire: ",
        pprint.pprint(self.rep)
        print "Last Move: " + str(self.lastMove)
        print "History: "
        print "  Rounds: ",
        pprint.pprint(self.historyRounds)
        print "  Moves: ",
        pprint.pprint(self.historyMoves)
        print "  Acts: ",
        pprint.pprint(self.historyActs)
        print "  Payoffs: ",
        pprint.pprint(self.historyPayoffs)
        print "  Demes: ",
        pprint.pprint(self.historyDemes)
        print "Born: " + str(self.born)
        print "Rounds Alive: " + str(self.roundsAlive)
        print "Times observed: " + str(self.nObserved)
        print "Current Deme: " + str(self.currentDeme)
        print "Points Earned: " + str(self.pointsEarned)
        print "Number of Offspring: " + str(self.nOffspring)

# Initialize structures in model
fitness = [] # fitness landscape
for i in range(3):
    tmp = [round(2*random.expovariate(lambd)**2) for x in range(nact)]
    fitness.append(tmp)
aliveAgents = []
Agents = []

outputFH.write("generation,strategy,nAgents,nInnovate,nObserve,nExploit,nRefine,totalPayoffs,avgLifespan,stdLifespan,medLifespan,maxLifespan\n")

# Initialize stats
class statsDict:
    Name = "statsDict"

    def __init__(self):
        self.aliveAgents = 0
        self.innovate = 0
        self.observe = 0
        self.exploit = 0
        self.refine = 0
        self.totalPayoffs = 0
        self.lifespans = []

    def report(self):
        outputline = str(self.aliveAgents)+","
        outputline += str(self.innovate)+","
        outputline += str(self.observe)+","
        outputline += str(self.exploit)+","
        outputline += str(self.refine)+","
        outputline += str(self.totalPayoffs)+","
        if len(self.lifespans)>0:
            outputline += str(mean(self.lifespans))+","
            outputline += str(std(self.lifespans))+","
            outputline += str(median(self.lifespans))+","
            outputline += str(max(self.lifespans))
        else:
            outputline += "0,0,0,0"
        return outputline


gameStats = []

# Initialize agents
if runIn:
    initStrategy = random.choice(strategies)
    for i in range(N):
        thisagent = newagent()
        Agents.append(thisagent)  # list of all agents
        Agents[i].strategy = initStrategy
        aliveAgents.append(i) # list of currently playing agents
else:
    for i in range(N):
        initStrategy = random.choice(strategies)
        thisagent = newagent()
        Agents.append(thisagent)  # list of all agents
        Agents[i].strategy = initStrategy
        aliveAgents.append(i) # list of currently playing agents


# Loop through each generation
for generation in range(ngen): 
    # initialize stats for this round
    roundStats = {}
    for strategy in strategies:
        roundStats[strategy] = statsDict()

    # create this round's data to pass to observe_who function
    if canChooseModel:
        modelInfo = []
        for i in aliveAgents:
            modelInfo.append([Agents[i].roundsAlive,
                             Agents[i].pointsEarned,
                             Agents[i].nObserved,
                             Agents[i].nOffspring])

    # Calculate total mean lifetime payoff for reproduction
    totalMeanPayoff = 0
    # Loop through each agent
    for i in aliveAgents:
	# teststring = globals()[Agents[i].strategy].test("a test signal")
        roundStats[Agents[i].strategy].aliveAgents += 1
	# get strategy from agent (call function "move" from imported strategy)
	theMove = globals()[Agents[i].strategy].move(Agents[i].roundsAlive, Agents[i].rep, Agents[i].historyRounds, Agents[i].historyMoves, Agents[i].historyActs, Agents[i].historyPayoffs, Agents[i].historyDemes, Agents[i].currentDeme, canChooseModel,canPlayRefine,multipleDemes)
        agentMove = theMove[0]
        if len(theMove) > 1:
            agentAction = theMove[1]
	# Do move:
        # INNOVATE
        if agentMove == INNOVATE:
            roundStats[Agents[i].strategy].innovate += 1
            unknownActs = set(range(100)) - set(Agents[i].rep.keys())
            if len(unknownActs) > 0:
                agentAction = random.choice(list(unknownActs))
                payoff = fitness[Agents[i].currentDeme][agentAction]
                Agents[i].rep[agentAction] = payoff

        # OBSERVE:
        elif agentMove == OBSERVE:
            agentAction = -1
            roundStats[Agents[i].strategy].observe += 1
            payoff = 0
            exploiters = []
            observed_acts = {}

            # can only observe agents in same deme who exploited on last round
            for j in aliveAgents:
                if j != i and Agents[j].lastMove==EXPLOIT and Agents[i].currentDeme==Agents[j].currentDeme:
                    exploiters.append(j)

            models = []
            # if observe_who extension is enabled
            if canChooseModel:
                # get ordered list of models (call observe_who function)
                rankedModels = globals()[Agents[i].strategy].observe_who(modelInfo)
                #TODO: handle no observe_who function
                #TODO: handle ranked models length < exploiters length
                for j in range(min(len(rankedModels),nObserve)):
                    if rankedModels[j] in exploiters:
                        models[j] = rankedModels[j]
            # else
            else:
                # select random group to observe from
                if nObserve < len(exploiters):
                    models = random.sample(exploiters,nObserve)
                else:
                    models = exploiters
                
            # create dictionary of actions & payoffs to add to repertoire
            for model in models:
                Agents[model].nObserved += 1
                for modelAction,modelPayoff in Agents[model].rep.iteritems():
                    if modelAction not in Agents[i].rep:
                        observed_acts[modelAction] = poisson.rvs(sigma,modelPayoff)
            
            # add copy errors
            for action in observed_acts:
                if random.random() < copy_error:
                    observed_acts[action] = 0

            # add to repertoire
            Agents[i].rep.update(observed_acts)
			
        # EXPLOIT
        elif agentMove == EXPLOIT:
            roundStats[Agents[i].strategy].exploit += 1
            if agentAction >= 0 and agentAction < 100:
                if agentAction in Agents[i].rep.keys():
                    payoff = fitness[Agents[i].currentDeme][agentAction]
                    Agents[i].pointsEarned += payoff
                    Agents[i].rep[agentAction] = payoff
                    roundStats[Agents[i].strategy].totalPayoffs += payoff
                else:
                    logger.error("Agent " + str(i) + " exploited action not in its repertoire")
            else:
                logger.error("Agent " + str(i) + " made an action less than 0 or grater than 100")
        
        # REFINE
        elif agentMove == REFINE and canPlayRefine:
            roundStats[Agents[i].strategy].refine += 1
            if agentAction in Agents[i].rep.keys():
                payoff = fitness[Agents[i].currentDeme][agentAction] + 1
                Agents[i].rep[agentAction] = payoff

        else:
            logger.error("Agent " + str(i) + " did not return a move or returned an invalid move: " + str(agentMove))
            continue

        # Write agent's move
        Agents[i].lastMove = agentMove
        Agents[i].historyRounds.append(generation)
        Agents[i].historyMoves.append(agentMove)
        Agents[i].historyActs.append(agentAction)
        Agents[i].historyPayoffs.append(payoff)
        Agents[i].historyDemes.append(Agents[i].currentDeme)

        # Write lifespan info
        Agents[i].roundsAlive += 1
        totalMeanPayoff += Agents[i].pointsEarned / Agents[i].roundsAlive
        roundStats[Agents[i].strategy].lifespans.append(Agents[i].roundsAlive)

    if totalMeanPayoff > 0:
        # Kill N random agents
        ndie = 0
        for i in range(len(aliveAgents)):
            if random.random() < pdie:
                ndie +=1
        random.shuffle(aliveAgents)
        del aliveAgents[0:ndie]

        # Choose N individuals to reproduce based on current relative fitness
        # get sum of average lifetime payoffs
        newAgents = []

        while len(newAgents) < ndie:
            i = random.choice(aliveAgents)
            if random.random() < (Agents[i].pointsEarned / Agents[i].roundsAlive)/totalMeanPayoff:
                if random.random() < pmut:
                    initStrategy = random.choice(strategies)
                else:
                    initStrategy = Agents[i].strategy
                thisagent = newagent()
                Agents.append(thisagent) 
                Agents[len(Agents)-1].strategy = initStrategy
                Agents[len(Agents)-1].born = generation
                newAgents.append(len(Agents)-1) # list of currently playing agents
        aliveAgents.extend(newAgents)

    ##TODO Write statistics for reproduction

    # Change environment
    for i in range(len(fitness)):
        for action in range(len(fitness[i])):
            if random.random() < pchg:
                fitness[i][action] = round(2*random.expovariate(lambd)**2)

    # Move agents if demes are enabled
    if multipleDemes:
        for i in aliveAgents:
            if random.random() < pmig:
                newdeme = [0,1,2].remove(Agents[i].currentDeme)
                Agents[i].currentDeme = random.choice(newdeme)
                
    for strategy in roundStats:
        fullline = str(generation)+","+strategy+","+roundStats[strategy].report()+"\n"
        outputFH.write(fullline)
    gameStats.append(roundStats)

outputFH.close()
##TODO Write summaries for entire run        
