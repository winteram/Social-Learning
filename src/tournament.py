# Model for Tournament for social learning strategies II contest
# author: Winter Mason (m@winteram.com)

# load required packages
import random
import sys, os
import logging
from scipy.stats import bernoulli,poisson,norm,expon,uniform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger=logging.getLogger('SLSTournament')

# Each strategy must have two functions, "makeMove" and "observe"
# "makeMove" takes as input: 
#    canPlayRefine, canChooseModel, currentDeme, roundsAlive, repertoire, history
# "makeMove"  must return: 
#    move, action, currentRep
# "observeWho" takes as input:
#    modelInfo, a list of dictionaries, one for each potential model, with the info:
#       age, total payoff, times observed, # offspring
# "observeWho" must return:
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

N = 100 # population size
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

    # create this round's data to pass to observeWho function
    if canChooseModel:
        modelInfo = []
        for i in aliveAgents:
            modelInfo[] = {"age":Agents[i]["roundsAlive"],
                           "totalPoints":Agents[i]["pointsEarned"],
                           "nObserved":Agents[i]["nObserved"],
                           "nOffspring":Agents[i]["nOffspring"]}

    # Loop through each agent
    for i in aliveAgents:
	# get strategy from agent (call function named "strategy")
	agentAction = globals()[Agents[i]["strategy"]].makeMove(canPlayRefine, canChooseModel,Agents[i]["currentDeme"], Agents[i]["roundsAlive"],Agents[i]["rep"],Agents[i]["history"])

        # Catch bad strategy output
        if move not in agentAction.keys():
            logger.error("Agent " + i + " did not return a move")
            continue

	# Do move:
        # INNOVATE
        if agentAction["move"] == "INNOVATE":
            gameStats[generation][Agents[i]["strategy"]]["innovate"] += 1
            unknownActs = set(payoff[Agents[i]["currentDeme"]].keys()) - set(Agents[i]["rep"].keys())
            if len(unknownActs) > 0:
                act = random.choice(unknownActs)
                payoff = payoff[Agents[i]["currentDeme"]][act]
                Agents[i]["rep"][act] = payoff

        # OBSERVE:
        elif agentAction["move"] == "OBSERVE":
            gameStats[generation][Agents[i]["strategy"]]["observe"] += 1
            payoff = 0
            exploiters = []
            observed_acts = {}

            # can only observe agents in same deme who exploited on last round
            for j in len(Agents):
                if j != i and Agents[j]["lastMove"]=="EXPLOIT" and Agents[i]["currentDeme"]==Agents[j]["currentDeme"]:
                    exploiters.append(j)

            models = []
            # if observe_who extension is enabled
            if canChooseModel:
                # get ordered list of models (call observeWho function)
                rankedModels = globals()[Agents[i]["strategy"]].observeWho(modelInfo)
                #TODO: handle no observeWho function
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
        elif agentAction["move"] == "EXPLOIT":
            gameStats[generation][Agents[i]["strategy"]]["exploit"] += 1
            if agentAction["action"]>0 and agentAction["action"] < 100:
            if agentAction["action"] in agentAction["rep"].keys():
                payoff = payoff[Agents[i]["currentDeme"]][agentAction["action"]] + 1
                Agents[i]["pointsEarned"] += payoff
                Agents[i]["rep"][agentAction["action"]] = payoff
                gameStats[generation][Agents[i]["strategy"]]["totalPayoffs"] += payoff
            else:
                logger.error("Agent " + i + " exploited action not in its repertoire")
        
        # REFINE
        elif agentAction["move"] == "REFINE" and canPlayRefine:
            gameStats[generation][Agents[i]["strategy"]]["refine"] += 1
            if agentAction["action"] in agentAction["rep"].keys():
                payoff = payoff[Agents[i]["currentDeme"]][agentAction["action"]] + 1
                agentAction["rep"][agentAction["action"]] = payoff

        # Write agent's move
        Agents[i]["lastMove"] = agentAction["move"]
        Agents[i]["history"]["historyRounds"][] = generation
        Agents[i]["history"]["historyMoves"][] = agentAction["move"]
        Agents[i]["history"]["historyActs"][] = agentAction["action"]
        Agents[i]["history"]["historyPayoffs"][] = payoff
        Agents[i]["history"]["historyDemes"][] = Agents[i]["currentDeme"]

        # Write lifespan info
        gameStats[generation][Agents[i]["strategy"]]["lifespans"][] = Agents[i]["roundsAlive"]        

    # Kill N random agents
    aliveAgents.shuffle()
    aliveAgents.slice(-Math.floor(pdie*len(AliveAgents))) # CHECK SYNTAX

    # Choose N individuals to reproduce based on current relative fitness
    for i in aliveAgents:
        if random.random() < Agents[i]["total"] / Agents[i]["roundsAlive"]:
            # reproduce

    # Randomly mutate x% of N new births

    # Write statistics for reproduction

    # Change environment

    # Move agents if demes are enabled

# Write summaries for entire run
