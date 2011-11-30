# Model for Tournament for social learning strategies II contest
# author: Winter Mason (m@winteram.com)

# load required packages
import random
import sys, os
import logging
from scipy.stats import bernoulli,poisson,norm,expon,uniform
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
strategies = []
for file in agentfiles:
    fname = file.split('.')
    if fname[1] == "py":
        import fname[0]
        strategies[] = fname[0]
##TODO: catch if no directory or files



##TODO: Allow importing of parameters with input file
# Initialize parameters in model
canPlayRefine = True  # if refine move is available
canChooseModel = True # if observe_who is an option
multipleDemes = True # if spatial extension is enabled
runIn = False # If there is a run-in time for the agents to develop

N = 100 # initial population size
ngen = 10000 # number of generations
nact = 100 # number of possible actions

lambd = 0.8 # 1 / mean of exponential distribution
pchg = 0.05 # probability of environment changing
pmut = 0.02 # probability of mutation
pmig = 0.03 # probability of migration (if spatial extension is enabled)
pdie = 0.02 # probability of dying

copy_error = 0.01
sigma = 0.2

# structure for a new agent
newagent = {"strategy":initStrategy, 
            "rep":{}, 
            "lastMove": -1,
            "history": {"historyRounds":[],
                        "historyMoves":[],
                        "historyActs":[],
                        "historyPayoffs":[], 
                        "historyDemes":[] }, 
            "born": 0,
            "roundsAlive": 0, 
            "nObserved":0,
            "currentDeme": 0, 
            "pointsEarned": 0,
            "nOffspring":0}

# Initialize structures in model
payoff = [] # payoff landscape
payoff[0] = [round(2*random.expovariate(lamd)**2) for x in range(nact)]
payoff[1] = [round(2*random.expovariate(lamd)**2) for x in range(nact)]
payoff[2] = [round(2*random.expovariate(lamd)**2) for x in range(nact)]


# Initialize stats
statsDict = {"aliveAgents":0,
             "innovate":0,
             "observe":0,
             "exploit":0,
             "refine":0,
             "totalPayoffs":0,
             "lifespans":[]}
roundStats = {}
for i in strategies:
    roundStats[strategies[i]] = statsDict
gameStats = []

# Initialize agents
if runIn:
    initStrategy = random.choice(strategies)
    for i in range(N):
        Agents[] = newagent  # list of all agents
        aliveAgents[] = i # list of currently playing agents
else:
    for i in range(N):
        initStrategy = random.choice(strategies)
        Agents[] = newagent  # list of all agents
        aliveAgents[] = i # list of currently playing agents


# Loop through each generation
for generation in range(ngen): 
    # initialize stats for this round
    gameStats[] = roundStats

    # create this round's data to pass to observe_who function
    if canChooseModel:
        modelInfo = []
        for i in aliveAgents:
            modelInfo[] = (Agents[i]["roundsAlive"],
                           Agents[i]["pointsEarned"],
                           Agents[i]["nObserved"],
                           Agents[i]["nOffspring"])

    # Loop through each agent
    for i in aliveAgents:
	# get strategy from agent (call function "move" from imported strategy)
	agentMove, agentAction = globals()[Agents[i]["strategy"]].move(Agents[i]["roundsAlive"], 
                                                                       Agents[i]["rep"], 
                                                                       Agents[i]["history"]["historyRounds"], 
                                                                       Agents[i]["history"]["historyMoves"], 
                                                                       Agents[i]["history"]["historyActs"], 
                                                                       Agents[i]["history"]["historyPayoffs"], 
                                                                       Agents[i]["history"]["historyDemes"], 
                                                                       Agents[i]["currentDeme"],
                                                                       canChooseModel,
                                                                       canPlayRefine,
                                                                       multipleDemes)
	# Do move:
        # INNOVATE
        if agentMove == INNOVATE:
            gameStats[generation][Agents[i]["strategy"]]["innovate"] += 1
            unknownActs = set(payoff[Agents[i]["currentDeme"]].keys()) - set(Agents[i]["rep"].keys())
            if len(unknownActs) > 0:
                act = random.choice(unknownActs)
                payoff = payoff[Agents[i]["currentDeme"]][act]
                Agents[i]["rep"][act] = payoff

        # OBSERVE:
        elif agentMove == OBSERVE:
            gameStats[generation][Agents[i]["strategy"]]["observe"] += 1
            payoff = 0
            exploiters = []
            observed_acts = {}

            # can only observe agents in same deme who exploited on last round
            for j in aliveAgents:
                if j != i and Agents[j]["lastMove"]==EXPLOIT and Agents[i]["currentDeme"]==Agents[j]["currentDeme"]:
                    exploiters.append(j)

            models = []
            # if observe_who extension is enabled
            if canChooseModel:
                # get ordered list of models (call observe_who function)
                rankedModels = globals()[Agents[i]["strategy"]].observe_who(modelInfo)
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
                Agents[model]["nObserved"] += 1
                for action,payoff in Agents[model]["rep"]:
                    if action not in Agents[i]["rep"]:
                        observed_acts[action] = poisson.rvs(sigma,payoff)
            
            # add copy errors
            for action in observed_acts:
                if random.random() < copy_error:
                    observed_acts[action] = 0

            # add to repertoire
            Agents[i]["rep"].extend(observed_acts) 
			
        # EXPLOIT
        elif agentMove == EXPLOIT:
            gameStats[generation][Agents[i]["strategy"]]["exploit"] += 1
            if agentAction>0 and agentAction < 100:
            if agentAction in agentAction["rep"].keys():
                payoff = payoff[Agents[i]["currentDeme"]][agentAction] + 1
                Agents[i]["pointsEarned"] += payoff
                Agents[i]["rep"][agentAction] = payoff
                gameStats[generation][Agents[i]["strategy"]]["totalPayoffs"] += payoff
            else:
                logger.error("Agent " + i + " exploited action not in its repertoire")
        
        # REFINE
        elif agentMove == REFINE and canPlayRefine:
            gameStats[generation][Agents[i]["strategy"]]["refine"] += 1
            if agentAction in Agents[i]["rep"].keys():
                payoff = payoff[Agents[i]["currentDeme"]][agentAction] + 1
                agentAction["rep"][agentAction] = payoff

        else:
            logger.error("Agent " + str(i) + " did not return a move or returned an invalid move: " + str(agentMove))
            continue

        # Write agent's move
        Agents[i]["lastMove"] = agentMove
        Agents[i]["history"]["historyRounds"][] = generation
        Agents[i]["history"]["historyMoves"][] = agentMove
        Agents[i]["history"]["historyActs"][] = agentAction
        Agents[i]["history"]["historyPayoffs"][] = payoff
        Agents[i]["history"]["historyDemes"][] = Agents[i]["currentDeme"]

        # Write lifespan info
        gameStats[generation][Agents[i]["strategy"]]["lifespans"][] = Agents[i]["roundsAlive"]        

    # Kill N random agents
    aliveAgents.shuffle()
    aliveAgents.slice(-Math.floor(pdie*len(AliveAgents))) # CHECK SYNTAX

    # Choose N individuals to reproduce based on current relative fitness
    # get sum of average lifetime payoffs
    for i in aliveAgents:
        if random.random() < Agents[i]["total"] / Agents[i]["roundsAlive"]:
            # reproduce

    # Randomly mutate x% of N new births

    # Write statistics for reproduction

    # Change environment

    # Move agents if demes are enabled

# Write summaries for entire run
