# Model for Tournament for social learning strategies II contest
# author: Winter Mason (m@winteram.com)

# load required packages
import random
import sys, os

# Load agents from agent directory
agent_dir = 'agents/'
agentfiles = os.listdir(agent_dir)
strategies = []
for file in agentfiles:
    fname = file.split('.')
    if fname[1] == "py":
        import fname[0]
        strategies[] = fname[0]

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
pmut = 0.04 # probability of mutation
pmig = 0.03 # probability of migration (if spatial extension is enabled)

copy_error = 0.01
sigma = 0.2

# structure for a new agent
newagent = {"strategy":initStrategy, "rep":{}, "lastmove":-99, "history":{"historyRounds":[],"historyMoves":[],"historyActs":[],"historyPayoffs":[], "historyDemes":[]}, "roundsAlive": 0 , "currentDeme": 0, "pointsEarned": 0}

# Initialize structures in model
payoff = [] # payoff landscape
payoff[0] = [round(2*random.expovariate(lamd)**2) for x in range(nact)]
payoff[1] = [round(2*random.expovariate(lamd)**2) for x in range(nact)]
payoff[2] = [round(2*random.expovariate(lamd)**2) for x in range(nact)]

if runIn:
    initStrategy = random.choice(strategies)
    for i in range(N):
        agentList[] = newagent # list of all agents
        aliveAgents[] = newagent  # list of currently playing agents
else:
    for i in range(N):
        initStrategy = random.choice(strategies)
        agentList[] =  # list of all agents
        aliveAgents[] = newagent  # list of currently playing agents


# Loop through each generation
for generation in range(ngen): 
    # Loop through each agent
    for i in range(N):
	# get strategy from agent (call function named "strategy")
	move, currentRep, observe_who = globals()[aliveAgents[i]["strategy"]](aliveAgents[i]["roundsAlive"],aliveAgents[i]["rep"],aliveAgents[i]["history"])
	# Do move:
        # if innovate, pick unknown behavior
        if move == -1:
            unknownActs = set(payoff[aliveAgents[i]["currentDeme"]]) - set(aliveAgents[i]["rep"].keys())
            if len(unknownActs) == 0:
                act = 0
                payoff = 0
            else:
                act = random.choice(unknownActs)
                payoff = payoff[aliveAgents[i]["currentDeme"]][act]
                aliveAgents[i]["rep"][act] = payoff
            aliveAgents[i]["lastmove"] = -1
            aliveAgents[i]["history"]["historyRounds"][] = generation
            aliveAgents[i]["history"]["historyMoves"][] = -1
            aliveAgents[i]["history"]["historyActs"][] = act
            aliveAgents[i]["history"]["historyPayoffs"][] = payoff
            aliveAgents[i]["history"]["historyDemes"][] = aliveAgents[i]["currentDeme"]

        # if observe:
        elif move == 0:
            exploiters = []
            observed_acts = {}
            # if there is anyone to observe
            for j in len(aliveAgents):
                if j != i and aliveAgents[j]["lastmove"]>0:
                    exploiters.append(j)

            models = []
            # if observe_who extension is enabled
            if canChooseModel:
                for model in range(min(len(exploiters),nObserve)):
                    if observe_who[model] in exploiters:
                        models[model] = observe_who[model]
            # else
            else:
                # select random group to observe from
                if nObserve < len(exploiters):
                    models = random.sample(exploiters,nObserve)
                else:
                    models = exploiters
                
            # create dictionary of actions & payoffs to add to repertoire
            for model in models:
                for action,payoff in aliveAgents[model]["rep"]:
                    if action not in aliveAgents[i]["rep"]:
                        observed_acts[action] = payoff + random.gauss(0,sigma)
            
            # add copy errors
            for action in observed_acts:
                if random.rand() < copy_error: # SYNTAXERROR
                    observed_acts[action] = 0

            # add to repertoire
            aliveAgents[i]["rep"].extend(observed_acts)  # SYNTAXERROR
			
        # if exploit, add payoff to agent's total
        elif move>0 and move < 100:
            aliveAgents[i]["pointsEarned"] += payoff[aliveAgents[i]["currentDeme"]][move]
        
        # if refine
        elif move==-2 and canPlayRefine:
            

        # Write agent's move
            
	# Write summary statistics for round

# Kill N random agents
            

	# Choose N individuals to reproduce based on current relative fitness

	# Randomly mutate x% of N new births

	# Write statistics for reproduction

	# Change environment

        # Move agents if demes are enabled

# Write summaries for entire run
