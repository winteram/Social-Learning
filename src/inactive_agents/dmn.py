#!/usr/bin/python

# discountmachine
# 
# submitted and coded by Daniel Cownden and Timothy Lillicrap

from numpy import *

def move(roundsalive,repertoire,historyR,historyM,historyA, historyP, historyDemes, currentDeme, canChooseModel, canPlayRefine, multipleDemes):
#def discountmachine(history_R, history_M, history_A, history_P, roundsalive, myrepertoire):

 	repertoire_idx = arange(100)
	repertoire_val = zeros(100)
# 	historyM = zeros(100)
# 	historyA = zeros(100)
# 	historyP = zeros(100)
	
 	for key, value in repertoire.items():
 		repertoire_val[key] = value
		
# 	for key, value in history_M.items():
# 		historyM[key] = value

# 	for key, value in history_A.items():
# 		historyA[key] = value
	
# 	for key, value in history_P.items():
# 		historyP[key] = value
	
	myrepertoire = vstack((repertoire_idx,repertoire_val))

	#put myhistory together as round, move, act, payoff
	myhistory = vstack((historyR, historyM, historyA, historyP))

	# declare the matrices used for our neural network decision function:
	bigmeans = array([[2.962799288818649], [0.266346085850312], [0.565819905341263]])
	bigwhiteningmatrix = array([[0.591050957984520, 0.055005581830927, 0.076760200599483], [0.055005581830927, 1.725797464539457, -0.260384199960065], [0.076760200599483, -0.260384199960065, 2.449235181316233]]);
	
	biglayers1_hidbiases = array([0.483281902800934, -0.802795653267638, -4.751792086448611,  0.501318962061778, -0.242300167488407,  1.119253225588153,  0.513474550233360, -0.266465646919917, -0.229735960254805, -0.931348556492638,  0.503058554055942,  0.252076332744831, -0.227013624507973,  3.158331613426219,  0.080032995999456])
	biglayers1_vishid = array([[-0.834713933833528, 0.380149763469969, -0.694685628199529, 0.146987282958356, 1.255440862567449, 0.802849982673830, -0.848904686787026, -0.264915151841741, 0.231134549352842, 0.866678522055704, 0.345018916948645, 0.534170855922578, 0.247036361970839, -0.200748504005506, 1.022725535776839], [0.194313533429005, 0.785679836935035, 4.011859032162985, -0.441495638638487, -1.640275366925572, 0.306533375147130, 0.574542808144203, 1.435776263444897, -0.904424098548293, 0.732400101179494, 0.200282919864061, 0.027496074324751, -0.938872880651827, -3.607943655198037, -3.027977767950190], [-0.350938843324459, -0.283800295606634, -1.640537237070937, 0.165001619132119, -1.225048656077579, -0.229936454765944, -0.614150093315454, -0.324911785356662, -0.039749519746621, -1.792106303325150, 1.461619538453021, -0.012379427034220, -0.048760310208360, 1.932285680317659, -2.083557783861354]])
	
	biglayers2_hidbiases = array([0.220209894558497])
	biglayers2_vishid = array([[1.099406342368920], [-1.357603333227983], [-2.745094030318414], [0.043960945983925], [1.874363888965300], [1.457579880366719], [0.930742101532256], [-1.112699142350296], [0.206064688682848], [0.927132526875494], [1.252498825504441], [-0.344394719981519], [0.229058886489332], [-2.742084831902555], [-2.136517480137703]])
	biglayers2_hidsums = array([[2.099406342368920], [-1.357603333227983], [-2.745094030318414], [0.043960945983925], [1.874363888965300], [1.457579880366719], [0.930742101532256], [-1.112699142350296], [0.206064688682848], [0.927132526875494], [1.252498825504441], [-0.344394719981519], [0.229058886489332], [-2.742084831902555], [-2.136517480137703]])

	
	
	if roundsalive == 0:
		#move = 0
		return (0,)
	elif roundsalive == 1 and sum(myrepertoire[1,:])==0:
		#move = -1
		return (-1,)
	else:
		if myhistory.size:
			nobs = sum(myhistory[1,:]==1)
		else:
			nobs = 1
		
		# calculate a scaling factor, the machine learned decision function
		# uses scaled values to be robust with respect to varying simulation
		# parameters
		scalefactor = max(myhistory[3,:])
		if scalefactor == 0:
			scalefactor = 1
		
		# estimate psubc and the mean of the payoff distribution and the number
		# of data points used in those estimates
		d_multiestimate = multiestimate(myrepertoire, myhistory, roundsalive)
		hatpsubc = d_multiestimate['hatpsubc']
		hatpsubcn = d_multiestimate['hatpsubcn']
		hatmean = d_multiestimate['hatmean']
		hatmeann = d_multiestimate['hatmeann']
		
		# calculate the average payoff observed and the number of observations 
		# made. this is used to estimate the value of observing
		hatobservablemean = mean(myhistory[3,myhistory[1,:]==0])
		hatobservablemeann = sum(myhistory[2,:]==0)
		
		# make this hatobservablemean robust to strange inputs
		if hatobservablemean == 0:
			hatobservablemean = hatmean
		
		# pair up observed payoffs with with the associated exploitpayoffs if
		# they exist. fit a line to these points and record the slope,
		# intercept and r^2 of the line along with the number of points used.
		d_linebuddy = linebuddy(myhistory, hatpsubc, scalefactor)
		slope = d_linebuddy['slope']
		intercept = d_linebuddy['intercept']
		npoints = d_linebuddy['npoints']
		rsquared = d_linebuddy['rsquared']
		
		# estimate what the best payoff is for the exploits in the repertoire
		# and return the best move and its associated payoff 
		d_estimatebestmoveandpayoff = estimatebestmoveandpayoff(roundsalive, myhistory, myrepertoire, hatpsubc, hatmean)
		hatbestmove = d_estimatebestmoveandpayoff['hatbestmove']
		hatbestpayoff = d_estimatebestmoveandpayoff['hatbestpayoff']
		
		
		# do the scaling, this scaling is only necisary for the machine learned
		# decision function.
		scaledhatbestpayoff = hatbestpayoff/scalefactor
		scaledhatobservablemean =  hatobservablemean/scalefactor
		
		# discountfactor is the chnace that next turn you will be alive and a 
		# given exploit that you were interested in has not changed its value
		discountfactor = ((1-hatpsubc)*(1-0.02))
		
		# we used the closed form of a geometric serries to calculate the total
		# points for exploiting the best thing in your repertoire until it 
		# changed or you died and the total points for observing (and thus 
		# missing a round of payoff) and then exploiting whatever you observed 
		# until it changed or you died.
		discountedhatobservablemean = scaledhatobservablemean*(discountfactor/(1-discountfactor))
		discountedhatbestpayoff = scaledhatbestpayoff*(1/(1-discountfactor))
		
		# a heuristic to make our creature more effective in low psubc
		# environmens.  It essentially causes the creature to occasionally look
		# at what others are doing if nothing has changed in a while, this way
		# the creature does not miss out on some fantastically high payoff 
		if hatpsubc < 0.05:
			if len(myhistory[1,:]) > 20:
				tmp = myhistory[1,(len(myhistory[1,:])-20):len(myhistory[1,:])] == myhistory[1,len(myhistory[1,:])-1]
				if sum(tmp) == 20:
					voodoo = 3 + sum(myhistory[3,myhistory[1,:]>0])/roundsalive
					if hatbestpayoff < voodoo or hatbestmove==0:
						#move = 0
						return (0,)
					else:
						#move = hatbestmove
						return (1,int(hatbestmove))

		# the network developed a pathology for this case.  so we use our old
		# simple minded decision criterion instead.
		if hatpsubc < 0.075 and hatpsubcn > 12:
			# the network has trouble with high nobserve in low psubc environments, 
			# so we use the old simple minded decision function again.
			if nobs > 3:
				if discountedhatbestpayoff < discountedhatobservablemean or hatbestmove==0:
					#move = 0
					return (0,)
				else:
					#move = hatbestmove
					return (1,int(hatbestmove))

			if npoints < 2:
				# the simple decision function we used before learning a neural network
				# (i.e. this way is based on theory we developed before tweaking)
				# we simply look at the difference between the estimated
				# discounted advantage of observing versus exploiting our best.
				if discountedhatbestpayoff < discountedhatobservablemean or hatbestmove==0:
					#move = 0
					return (0,)
				else:
					#move = hatbestmove
					return (1,int(hatbestmove))

		# big black box decision function here:
		grandstackable = array([nobs, slope, rsquared, discountedhatbestpayoff, discountedhatobservablemean])
		
		# we feed grandstackable to a machine learned decision function that
		# takes into account how the value of observing my be altered by both
		# nobs and by the reliability of observing.  We trained this function
		# by having it try and match the estimate made by a creature with
		# perfect knowledge of what could be observed and P_actionnoise.
		# CALL BBB (MAKE SURE DECISION TUPLE IS A GOOD IDEA)
		d_bbb = bbb(grandstackable, bigmeans, bigwhiteningmatrix, biglayers1_hidbiases, biglayers1_vishid, biglayers2_hidbiases, biglayers2_vishid, biglayers2_hidsums)
		decision = d_bbb['decision']
		
		if array_equal(decision, array([1, 0])) or hatbestmove==0:
			#move = 0
			return (0,)
		elif array_equal(decision, array([0, 1])):
			#move = hatbestmove
			return (1,int(hatbestmove))

		
	# end main if statements
# END MAIN FUNCTION

def estimatebestmoveandpayoff(roundsalive, myhistory, myrepertoire, hatpsubc, hatmean):

	# crawl through the reperoire making sure it is properly updated:
	for idx, val in list(enumerate(myrepertoire[0])):
		
		# everything from the history associated with this given exploit
		if myhistory[:,myhistory[2,:] == myrepertoire[0,idx]].shape[1] > 0:
			pertinent = myhistory[:,myhistory[2,:] == myrepertoire[0,idx]]
		else:
			pertinent = array([[0],[0],[-1],[0]])
		
		# if the most recent information about this exploit is from an observe

		if pertinent[2,-1] != 0:
			# if the most recent infromation is from an exploit or an innovate put that in the repertoire
			myrepertoire[1,idx] = pertinent[3,-1]
		else:
			d_observationroller = observationroller(pertinent, hatpsubc)
			obsrollin = d_observationroller['obsrollin']
			myrepertoire[1,idx] = obsrollin
		tmp = pertinent[0,:]
		tmp = tmp[-1]
		timesincelast = roundsalive - tmp
		
		# chance of no change = (1-hatpsubc)^timesincelast;
		# here we calculate the expected value of each action in the
		# repertoire by doing a weighted average between the previous
		# observed value and our estimate of the mean, with the weights
		# determined by our estimate of psubc and the time since this
		# exploit was last exploited or observed
		myrepertoire[1,idx] = (1-pow((1-hatpsubc),timesincelast)) * hatmean + (pow((1-hatpsubc),timesincelast) * myrepertoire[1,idx])
		
		# then we pick the one with the best expected payoff and pass that out
		hatbestpayoff = max(myrepertoire[1,:])
		# this tmp business makes the selction process robust to multiple
		# equally valued exploits
		tmp = myrepertoire[0,myrepertoire[1,:] == hatbestpayoff]
		hatbestmove = tmp[0]
		
		return dict({'hatbestmove': hatbestmove, 'hatbestpayoff': hatbestpayoff})
# END ESTIMATEBESTMOVEANDPAYOFF FUNCTION

def multiestimate(myrepertoire, myhistory, roundsalive):
	# in the first few rounds of life don't bother trying to calculate
	# psubc just guess a safe default and use whatever is around to
	# estimate the mean
	if roundsalive < 3:
		hatpsubc = 0.001
		hatpsubcn = 0
		hatmean = sum(myhistory[3,:])/len(myhistory[3,:])
		hatmeann = 0
		return dict({'hatpsubc': hatpsubc, 'hatpsubcn': hatpsubcn, 'hatmean': hatmean, 'hatmeann': hatmeann})
	else:
		# initialize some computaitonally useful variables 
		psubcnumerator = 0
		psubcdenomenator = 0
		meanpayoffnumerator = 0
		meanpayoffdenomenator = 0
		meanpayoffnumerator2 = 0
		meanpayoffdenomenator2 = 0
		#crawl through the repertoire
		
		for idx, val in enumerate(myrepertoire[0]):
			#calculates the mean payoff for that exploit using exploits and
			#observes
			sequencei = myhistory[3, myhistory[2,:] == myrepertoire[0,idx]]
			meanpayoffnumerator2 = meanpayoffnumerator2 + sum(sequencei)
			meanpayoffdenomenator2 = meanpayoffdenomenator2 + len(sequencei)


			# calculate the mean payoff and psubc without using observes
			if myhistory[:,myhistory[2,:] == myrepertoire[0,idx]].shape[1] > 0:
				pertinenti = myhistory[:, myhistory[2,:] == myrepertoire[0,idx]]
			else:
				pertinenti = array([[0],[0],[-1],[0]])
			pertinenti = pertinenti[:, pertinenti[1,:] != 0]
			pertinenti = pertinenti[array([0,3]),:]
			
			if len(pertinenti[0]) > 1:
				#if there are multiple payoffs associated with that action
				# we use the diff operator to note when the payoff for a
				# given exploit has changed
				differences = diff(pertinenti,1,1)
				difflogicalindex = differences[1,:] != 0
				
				# what the distinct values are
				meanpayoffnumerator = meanpayoffnumerator + pertinenti[1,0]
				# what the distinct values are
				meanpayoffnumerator = meanpayoffnumerator + dot(pertinenti[1,1:len(pertinenti[1])],difflogicalindex)

				# the number of distinct values
				meanpayoffdenomenator = meanpayoffdenomenator + 1 + sum(difflogicalindex)

				# capping the differences at 12 is a fudge factor to
				# prevent an underestimation error of psubc
				tmp = differences[0,differences[1,:] == 0]
				tmp[tmp > 12] = 12

				# a sum of the time between exploits when there was no change in the value with that time capped of at 12
				psubcdenomenator = psubcdenomenator + sum(tmp)
				
				# a count of all the times the value of the exploit did change
				psubcnumerator = psubcnumerator + len(differences[1,difflogicalindex-1])
				#########################################################################################################################################
				psubcdenomenator = psubcdenomenator + psubcnumerator
			# if there is only one exploit associated with the action
			elif len(pertinenti[0]) == 1:
				# add that one exploit to the mean estimate
				meanpayoffnumerator = meanpayoffnumerator + pertinenti[1,0]
				meanpayoffdenomenator = meanpayoffdenomenator + 1;
				# we can't deduce anything about psubc from just one exploit

	# make the estimator robust to extreme cases
	if psubcdenomenator == 0:
		hatpsubc = 0.001
		hatpsubcn = 0
	else:
		hatpsubcn = psubcdenomenator
		# we estimate psubc by taking the number of known chnages divided by (a close proxy of) the opportunities for change
		hatpsubc = psubcnumerator/psubcdenomenator
		# we know the range of psubc so we hard limit it
		if hatpsubc > 0.4:
			hatpsubc = 0.4
		if hatpsubc < 0.001:
			hatpsubc = 0.001
	
	# if your in the early stages of life
	if roundsalive < 6:
		# use observables in your estimate of the payoff mean
		hatmeann = meanpayoffdenomenator2
		hatmean = meanpayoffnumerator2/meanpayoffdenomenator2
	# or if you don't have any exploits
	elif meanpayoffdenomenator == 0:
		# use observables in your estimate of the payoff mean
		hatmeann = meanpayoffdenomenator2
		hatmean = meanpayoffnumerator2/meanpayoffdenomenator2
	# if your old and you've exploited
	else:
		# don't use observed values in your estimate of the payoff mean
		hatmeann = meanpayoffdenomenator
		hatmean = meanpayoffnumerator/meanpayoffdenomenator
	
	return dict({'hatpsubc': hatpsubc, 'hatpsubcn': hatpsubcn, 'hatmean': hatmean, 'hatmeann': hatmeann})
# END MULTIESTIMATE FUNCTION

def linebuddy(myhistory, hatpsubc, scalefactor):
	# gives a logical index of all the places in myhistory where an observe occured
	obsindices = myhistory[1,:] == 0
	linelist = (-1)*ones((2,sum(obsindices)));
	# a vector of the turns observes occured
	turn = myhistory[0, obsindices-1]
	# a vector of the actions observed
	action = myhistory[2, obsindices-1]
	# a vector of the payoffs observed
	payoff = myhistory[3, obsindices-1]
	
	# here we compute how far we are willing to look ahead for correlated
	# exploits, this is a psubc dependent range of search 
	#(0.9 threshhold, accepts data that is corect this often 90# of the time)
	nrange = ceil(log(0.9)/(log(1-hatpsubc)))
	
	# for each observation made
	for i in range(0, sum(obsindices)):
		# see which part of the history is in range
		inrange = myhistory[:, myhistory[0,:] >= turn[i]]
		inrange = inrange[:, inrange[0,:] <= (turn[i] + nrange)]
		
		# see if there are any exploits in that range
		exploits = inrange[:, inrange[1,:] == action[i]]
		# if there were exploits
		if exploits.size:
			# note the value of the exploit
			expvalue = exploits[3,0]
			# the observed values are in the first row
			# the correlated exploit values are in the row below
			# form a data point
			linelist[:,i] = [payoff[i],expvalue-1]

	# drop all the observes that didn't have a correlated exploit
	linelist = linelist[:,linelist[1,:]>=0]
	# scale everything, this make the line fit robust to extreme values
	linelist = (linelist / float(scalefactor))
	
	# if we only have one data point
	if len(unique(linelist[0,:])) < 2:
		# make this safe default guess
		npoints = 0
		slope = pi/4
		intercept = 0
		rsquared = 1
	# we have two or more data points
	else:
		# fit the line
		X = ones((len(linelist[0,:]),2))
		X[:,0]=linelist[0,:]
		Y = linelist[1,:]
		beta = asarray(linalg.inv(transpose(mat(X))*X)*(transpose(mat(X))*transpose(mat(Y))))
		# we hope that slope along with rsquared gives some idea about how high
		# or low P_actionnoise is
		# we take the arctan of the slope so that the machine learned function
		# has nice bounded values to work with.
		slope = arctan(beta[0])
		intercept = beta[2]
		npoints = len(Y) - 1
		#sum of squared errors
		sse = sum(power((transpose(mat(beta))*transpose(mat(X)) - Y),2))
		# sum of total squared error
		sst = sum((Y - mean(Y))**2)
		if sst == 0:
			# the precentage of total variation explained by the line
			rsquared = 1
		# our line fitter breaks down for slopes near zero and this makes the it robust to this
		elif sst < sse:
			rsquared = 1
		else:
			# the precentage of total variation explained by the line
			rsquared = 1 - (sse/sst)
	
	return dict({'slope': slope, 'intercept': intercept, 'npoints': npoints, 'rsquared': rsquared})
# END LINEBUDDY FUNCTION

def	observationroller(pertinent, hatpsubc):
	#trim the data coming in
	if len(pertinent[0]) > 10:
		pertinent = pertinent[:,len(pertinent[0])-11:len(pertinent[0])]
	
	#if there has been an exploit prior to this most recent observation
	#then trim the data to include the most recent exploit and everything
	#after that
	tmp = nonzero(pertinent[1,:]!=0)
	tmp = tmp[0]
	if tmp.size:
		lastexploitindex = tmp[len(tmp)-1]
		pertinent = pertinent[:,lastexploitindex:len(pertinent[0])]
	
	# takes a weighted average of all the pertinent values.  It gives more
	# weight to the more recent values and how much it discounts old values
	# depends on what the estimate of psubc is.  Old data is no good in
	# high psubc environments but just fine in low psubcenvironments.
	times = pertinent[0,:]  
	values = pertinent[3,:]
	timediffs = pertinent[0,len(pertinent[0])-1] - times
	T = (1-hatpsubc)**timediffs
	Tdenomenator = sum(T);
	obsrollin = sum(values * transpose(mat(T)))/Tdenomenator;
	
	return dict({'obsrollin': obsrollin})
#END OBERSERVATIONROLLER FUNCTION

# we use a standard feedforward neural network to tweak the value of the 
# discounted scaled observables.  we trained the network by building a
# 'cheating' creature which had more knowledge than ours, and learned to
# copy this creature's behaviour.  the network is composed of standard 
# sigmoidal units. we used a version of conjugate gradient descent coupled 
# with line searches to optimize the weights in the network (this is just a
# very nice fast version of the backpropagation algorithm).  
# we also do standard preprocessing to the data (e.g. whitening).
def bbb(grandstackable, bigmeans, bigwhiteningmatrix, biglayers1_hidbiases, biglayers1_vishid, biglayers2_hidbiases, biglayers2_vishid, biglayers2_hidsums):
	# prepare the input data for the network (subtract means and whiten):
	data = grandstackable[0:3] - transpose(bigmeans)
	data = transpose(bigwhiteningmatrix * transpose(mat(data)))
	discountedhatbest = grandstackable[3]
	discountedobs = grandstackable[4]
	
	# forwards pass through the network:
	layers1_hidbiases = biglayers1_hidbiases
	layers1_vishid = biglayers1_vishid
	layers2_hidbiases = biglayers2_hidbiases
	layers2_vishid = biglayers2_vishid
	layers2_hidsums = biglayers2_hidsums
	numcases = 1

	layers1_hidsums = data * mat(layers1_vishid) + tile(layers1_hidbiases, (numcases, 1))
	layers1_hidacts = 1 / (1 + power(e, -layers1_hidsums))
	numlayers = 2
	
	## WAS A LOOP BUT WHY?
	layers2_hidsums = layers1_hidacts * layers2_vishid + tile(layers2_hidbiases, (numcases, 1))
	layers2_hidacts = 1 / (1 + power(e, -layers2_hidsums))
	## END OF POINTLESS LOOP?
	
	preguess = array([((2*layers2_hidacts[0,0])*discountedobs), discountedhatbest])

	decision = array([0,0])
	
	decision[0] = preguess[0] > preguess[1]
	decision[1] = preguess[0] <= preguess[1]
	
	return dict({'decision': decision});
#END BIGBLACKBOX BBB FUNCTION

def observe_who(exploiterData):
	return sorted(exploiterData,key=lambda x:x[AGE],reverse=True)
