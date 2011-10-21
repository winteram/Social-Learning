function [move myrepertoire] = discountmachine(roundsalive, myrepertoire, myhistory)
% discountmachine
% 
% submitted and coded by Daniel Cownden and Timothy Lillicrap

myhistory = double(myhistory); % ADDED BY L RENDELL AS MYHISTORY STORED AS INT16 IN SIMULATION

% declare the matrices used for our neural network decision function:
bigmeans = [2.962799288818649;   0.266346085850312;   0.565819905341263];
bigwhiteningmatrix =     [ 0.591050957984520   0.055005581830927   0.076760200599483;   0.055005581830927   1.725797464539457  -0.260384199960065;   0.076760200599483  -0.260384199960065   2.449235181316233];
biglayers{1}.hidbiases = [ 0.483281902800934  -0.802795653267638  -4.751792086448611   0.501318962061778  -0.242300167488407   1.119253225588153   0.513474550233360  -0.266465646919917  -0.229735960254805  -0.931348556492638   0.503058554055942   0.252076332744831  -0.227013624507973   3.158331613426219   0.080032995999456];
biglayers{1}.vishid =    [-0.834713933833528   0.380149763469969  -0.694685628199529   0.146987282958356   1.255440862567449   0.802849982673830  -0.848904686787026  -0.264915151841741   0.231134549352842   0.866678522055704   0.345018916948645   0.534170855922578   0.247036361970839  -0.200748504005506   1.022725535776839; ...
                           0.194313533429005   0.785679836935035   4.011859032162985  -0.441495638638487  -1.640275366925572   0.306533375147130   0.574542808144203   1.435776263444897  -0.904424098548293   0.732400101179494   0.200282919864061   0.027496074324751  -0.938872880651827  -3.607943655198037  -3.027977767950190; ...
                          -0.350938843324459  -0.283800295606634  -1.640537237070937   0.165001619132119  -1.225048656077579  -0.229936454765944  -0.614150093315454  -0.324911785356662  -0.039749519746621  -1.792106303325150   1.461619538453021  -0.012379427034220  -0.048760310208360   1.932285680317659  -2.083557783861354];
biglayers{2}.hidbiases = [0.220209894558497];
biglayers{2}.vishid = [1.099406342368920;  -1.357603333227983;  -2.745094030318414;   0.043960945983925;   1.874363888965300;   1.457579880366719;   0.930742101532256;  -1.112699142350296;   0.206064688682848;   0.927132526875494;   1.252498825504441;  -0.344394719981519;   0.229058886489332;  -2.742084831902555;  -2.136517480137703];

if (roundsalive == 0), % first turn:
    move = 0; % always start by trying to observe
elseif (and(roundsalive == 1, isempty(myrepertoire))), % second turn and an empty repertoire
    move = -1; % innovate if there was nothing to observe on the first round
else % for every other situation
    
    % estimate nobserve:
    if (~isempty(myhistory)),
        nobs = sum(myhistory(1,:)==1);
    else
        nobs = 1;
    end;
    
    % calculate a scaling factor, the machine learned decision function
    % uses scaled values to be robust with respect to varying simulation
    % parameters
    scalefactor = max(myhistory(4,:));
    if (scalefactor == 0),
        scalefactor = 1;
    end;
    
    % estimate psubc and the mean of the payoff distribution and the number
    % of data points used in those estimates
    [hatpsubc hatpsubcn hatmean hatmeann] = multiestimate(myrepertoire, myhistory, roundsalive);    
    
    % calculate the average payoff observed and the number of observations 
    % made. this is used to estimate the value of observing
    hatobservablemean = mean(myhistory(4,myhistory(2,:)==0));
    hatobservablemeann = sum(myhistory(2,:)==0);
    
    % make this hatobservablemean robust to strange inputs
    if (hatobservablemean == 0),
        hatobservablemean = hatmean;
    end;
    
    % pair up observed payoffs with with the associated exploitpayoffs if
    % they exist. fit a line to these points and record the slope,
    % intercept and r^2 of the line along with the number of points used.
    [slope intercept npoints rsquared] = linebuddy(myhistory, hatpsubc, scalefactor);
    
    % estimate what the best payoff is for the exploits in the repertoire
    % and return the best move and its associated payoff 
    [hatbestmove,hatbestpayoff] = estimatebestmoveandpayoff(roundsalive, myhistory, myrepertoire, hatpsubc, hatmean);
    
    % do the scaling, this scaling is only necisary for the machine learned
    % decision function.
    scaledhatbestpayoff = (hatbestpayoff/scalefactor);
    scaledhatobservablemean = (hatobservablemean/scalefactor);
    
    % discountfactor is the chnace that next turn you will be alive and a 
    % given exploit that you were interested in has not changed its value
    discountfactor = ((1-hatpsubc)*(1-0.02));
    
    % we used the closed form of a geometric serries to calculate the total
    % points for exploiting the best thing in your repertoire until it 
    % changed or you died and the total points for observing (and thus 
    % missing a round of payoff) and then exploiting whatever you observed 
    % until it changed or you died.
    discountedhatobservablemean = scaledhatobservablemean*(discountfactor/(1-discountfactor)); 
    discountedhatbestpayoff = scaledhatbestpayoff*(1/(1-discountfactor));

    % a heuristic to make our creature more effective in low psubc
    % environmens.  It essentially causes the creature to occasionally look
    % at what others are doing if nothing has changed in a while, this way
    % the creature does not miss out on some fantastically high payoff 
    if (hatpsubc < 0.05),
        if (length(myhistory(2,:)) > 20),
            tmp = myhistory(2,end-19:end) == myhistory(2,end);
            if (sum(tmp) == 20)
                voodoo = 3 + (sum(myhistory(4, (myhistory(2,:) > 0))) / roundsalive);
                if (hatbestpayoff < voodoo),
                    move = 0;
                else
                    move = hatbestmove;
                end;
                return;
            end;
        end;    
    end;

   % the network developed a pathology for this case.  so we use our old
   % simple minded decision criterion instead.
   if (and(hatpsubc < 0.075, hatpsubcn > 12))
       % the network has trouble with high nobserve in low psubc environments, 
       % so we use the old simple minded decision function again.
       if (nobs > 3),
           if (discountedhatbestpayoff < discountedhatobservablemean )
               move = 0;
           else
               move = hatbestmove;
           end;
           return;           
       end;
       if (npoints < 2),
           % the simple decision function we used before learning a neural network
           % (i.e. this way is based on theory we developed before tweaking)
           % we simply look at the difference between the estimated
           % discounted advantage of observing versus exploiting our best.
           if (discountedhatbestpayoff < discountedhatobservablemean )
               move = 0;
           else
               move = hatbestmove;
           end;
           return;
       end;
   end;
   
   % big black box decision function here:
    grandstackable = [nobs slope rsquared discountedhatbestpayoff discountedhatobservablemean];

    % we feed grandstackable to a machine learned decision function that
    % takes into account how the value of observing my be altered by both
    % nobs and by the reliability of observing.  We trained this function
    % by having it try and match the estimate made by a creature with
    % perfect knowledge of what could be observed and P_actionnoise.
    decision = bbb(grandstackable, bigmeans, bigwhiteningmatrix, biglayers);
    if (decision == [1 0])
        move = 0;
    elseif (decision == [0 1])
        move = hatbestmove;
    end;
    
end % main if statements

function [hatbestmove,hatbestpayoff] = estimatebestmoveandpayoff(roundsalive, myhistory, myrepertoire, hatpsubc, hatmean)
    
    % crawl through the reperoire making sure it is properly updated:
    for i = 1:length(myrepertoire(1,:))
        
        % everything from the history associated with this given exploit
        pertinent = myhistory(:,myhistory(3,:) == myrepertoire(1,i)); 
        
         % if the most recent information about this exploit is from an observe
        if (pertinent(2,end) == 0),
            obsrollin = observationroller(pertinent, hatpsubc); % see the function for a description of obsrollin
            myrepertoire(2,i) = obsrollin;
        else % if the most recent infromation is from an exploit or an innovate put that in the repertoire
            myrepertoire(2,i) = pertinent(4,end);
        end;
        tmp = pertinent(1,:);
        tmp = tmp(end);
        timesincelast = roundsalive - tmp;
        % chance of no change = (1-hatpsubc)^timesincelast;
        % here we calculate the expected value of each action in the
        % repertoire by doing a weighted average between the previous
        % observed value and our estimate of the mean, with the weights
        % determined by our estimate of psubc and the time since this
        % exploit was last exploited or observed
        myrepertoire(2,i) = (1-(1-hatpsubc)^timesincelast) * hatmean + ((1-hatpsubc)^timesincelast) * myrepertoire(2,i);
    end;

    % then we pick the one with the best expected payoff and pass that out
    hatbestpayoff = max(myrepertoire(2,:));
    % this tmp business makes the selction process robust to multiple
    % equally valued exploits
    tmp = myrepertoire(1,myrepertoire(2,:) == hatbestpayoff);
    hatbestmove = tmp(1);
    
end %estimatebestmoveandpayoff function    
end % main function

function [hatpsubc, hatpsubcn, hatmean, hatmeann] = multiestimate(myrepertoire, myhistory, roundsalive)
    
    % in the first few rounds of life don't bother trying to calculate
    % psubc just guess a safe default and use whatever is around to
    % estimate the mean
    if (roundsalive < 3), 
        hatpsubc = 0.001;
        hatpsubcn = 0;
        hatmean = sum(myhistory(4,:))/length(myhistory(4,:));
        hatmeann = 0;
        return;
    else
        % initialize some computaitonally useful variables 
        psubcnumerator = 0; psubcdenomenator = 0;
        meanpayoffnumerator = 0; meanpayoffdenomenator = 0;
        meanpayoffnumerator2 = 0; meanpayoffdenomenator2 = 0;
        %crawl through the repertoire
        
        for i = 1:length(myrepertoire(1,:)),
            
            %calculates the mean payoff for that exploit using exploits and
            %observes
            sequencei = myhistory(4, (myhistory(3,:) == myrepertoire(1,i)));
            meanpayoffnumerator2 = meanpayoffnumerator2 + sum(sequencei);
            meanpayoffdenomenator2 = meanpayoffdenomenator2 + length(sequencei);
            
            % calculate the mean payoff and psubc without using observes
            pertinenti = myhistory([1 4], and(myhistory(3,:) == myrepertoire(1,i), myhistory(2,:) ~= 0));
            
            if size(pertinenti,2) > 1, %if there are multiple payoffs associated with that action
                % we use the diff operator to note when the payoff for a
                % given exploit has changed
                differences = diff(pertinenti,1,2);
                difflogicalindex = (differences(2,:) ~= 0);
                
                meanpayoffnumerator = meanpayoffnumerator + pertinenti(2,1); % what the distinct values are
                meanpayoffnumerator = meanpayoffnumerator + sum(pertinenti(2,2:end)*difflogicalindex'); % what the distinct values are

                meanpayoffdenomenator = meanpayoffdenomenator + 1 + sum(difflogicalindex); % the number of distinct values

                % capping the differences at 12 is a fudge factor to
                % prevent an underestimation error of psubc
                tmp = differences(1,(differences(2,:) == 0));
                tmp(tmp > 12) = 12;

                psubcdenomenator = psubcdenomenator + sum(tmp); % a sum of the time between exploits when there was no change in the value with that time capped of at 12
                % psubcnumerator = psubcnumerator + size(differences(difflogicalindex),2); % a count of all the times the value of the exploit did change
                %CHANGED THIS LINE BECAUSE OCTAVE INTERPRETS LOGICAL INDICES INTO MATRICES A BIT DIFFERENTLY TO MATLAB
                psubcnumerator = psubcnumerator + size(differences(1,difflogicalindex),2); % a count of all the times the value of the exploit did change
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                psubcdenomenator = psubcdenomenator + psubcnumerator;
            elseif size(pertinenti,2) == 1, % if there is only one exploit associated with the action
                % add that one exploit to the mean estimate
                meanpayoffnumerator = meanpayoffnumerator + pertinenti(2,1);
                meanpayoffdenomenator = meanpayoffdenomenator + 1;
                % we can't deduce anything about psubc from just one exploit
            else
            end;
        end;
    end;
    
    if (psubcdenomenator == 0), % make the estimator robust to extreme cases
        hatpsubc = 0.001;
        hatpsubcn = 0;
    else
        hatpsubcn = psubcdenomenator;
        hatpsubc = (psubcnumerator/psubcdenomenator); % we estimate psubc by taking the number of known chnages divided by (a close proxy of) the opportunities for change
        if (hatpsubc > 0.4), hatpsubc = 0.4; end; % we know the range of psubc so we hard limit it
        if (hatpsubc < 0.001), hatpsubc = 0.001; end;        
    end;
    
    if (roundsalive < 6), % if your in the early stages of life
        % use observables in your estimate of the payoff mean
        hatmeann = meanpayoffdenomenator2;
        hatmean = meanpayoffnumerator2/meanpayoffdenomenator2;        
    elseif (meanpayoffdenomenator == 0), % or if you don't have any exploits
        % use observables in your estimate of the payoff mean
        hatmeann = meanpayoffdenomenator2;
        hatmean = meanpayoffnumerator2/meanpayoffdenomenator2;          
    else % if your old and you've exploited
        % don't use observed values in your estimate of the payoff mean
        hatmeann = meanpayoffdenomenator;
        hatmean = meanpayoffnumerator/meanpayoffdenomenator;
    end;
           
end % multiestimate

function [slope intercept npoints rsquared] = linebuddy(myhistory, hatpsubc, scalefactor)
obsindices = (myhistory(2,:) == 0); % gives a logical index of all the places in myhistory where an observe occured
linelist = (-1)*ones(2,sum(obsindices));
turn = myhistory(1, obsindices); % a vector of the turns observes occured
action = myhistory(3, obsindices); % a vector of the actions observed
payoff = myhistory(4, obsindices); % a vector of the payoffs observed

% here we compute how far we are willing to look ahead for correlated
% exploits, this is a psubc dependent range of search 
%(0.9 threshhold, accepts data that is corect this often 90% of the time)
nrange = ceil(log(0.9)/(log(1-hatpsubc)));

for i = 1:sum(obsindices), % for each observation made
    inrange = myhistory(:, and(myhistory(1,:) >= turn(i), myhistory(1,:) <= turn(i) + nrange)); % see which part of the history is in range
    exploits = inrange(:, inrange(2,:) == action(i)); % see if there are any exploits in that range
    if (~isempty(exploits)), % if there were exploits
        expvalue = exploits(4,1); % note the value of the exploit
        % the observed values are in the first row
        % the correlated exploit values are in the row below
        linelist(:,i) = [payoff(i),expvalue]; % form a data point
    end;
end;

linelist = linelist(:,linelist(2,:)>=0); % drop all the observes that didn't have a correlated exploit
linelist = (linelist ./ scalefactor) ; % scale everything, this make the line fit robust to extreme values
if (size(unique(linelist(1,:)),2) < 2), % if we only have one data point
    % make this safe default guess
    npoints = 0;
    slope = pi/4;
    intercept = 0;
    rsquared = 1;
else % we have two or more data points
    % fit the line
    X = ones(length(linelist(1,:)),2);
    X(:,1)=linelist(1,:);
    Y = linelist(2,:);
    beta = inv(X'*X)*(X'*Y');
    % we hope that slope along with rsquared gives some idea about how high
    % or low P_actionnoise is
    % we take the arctan of the slope so that the machine learned function
    % has nice bounded values to work with.
    slope = atan(beta(1));
    intercept = beta(2);
    npoints = length(Y) - 1;
    sse = sum((beta'*X' - Y).^2); %sum of squared errors
    sst = sum((Y - mean(Y)).^2); % sum of total squared error
    if (sst == 0),
        rsquared = 1; % the precentage of total variation explained by the line
    elseif (sst < sse), % our line fitter breaks down for slopes near zero and this makes the it robust to this
        rsquared = 1;
    else
        rsquared = 1 - (sse/sst); % the precentage of total variation explained by the line
    end;
end;
end % linebuddy

function [obsrollin] = observationroller(pertinent, hatpsubc)
    %trim the data coming in
    if (size(pertinent,2) > 10),
        pertinent = pertinent(:,end-10:end);
    end;
    
    %if there has been an exploit prior to this most recent observation
    %then trim the data to include the most recent exploit and everything
    %after that
    tmp = find(pertinent(2,:)~=0);
    if (~isempty(tmp))
        lastexploitindex = tmp(end);
        pertinent = pertinent(:,lastexploitindex:end);
    end;    
    
    % takes a weighted average of all the pertinent values.  It gives more
    % weight to the more recent values and how much it discounts old values
    % depends on what the estimate of psubc is.  Old data is no good in
    % high psubc environments but just fine in low psubcenvironments.
    times = pertinent(1,:);  
    values = pertinent(4,:);
    timediffs = pertinent(1,end) - times;
    T = (1-hatpsubc).^timediffs;
    Tdenomenator = sum(T);
    obsrollin = sum(values * T')/Tdenomenator;
end % observation roller

% we use a standard feedforward neural network to tweak the value of the 
% discounted scaled observables.  we trained the network by building a
% 'cheating' creature which had more knowledge than ours, and learned to
% copy this creature's behaviour.  the network is composed of standard 
% sigmoidal units. we used a version of conjugate gradient descent coupled 
% with line searches to optimize the weights in the network (this is just a
% very nice fast version of the backpropagation algorithm).  
% we also do standard preprocessing to the data (e.g. whitening).
function decision = bbb(grandstackable, bigmeans, bigwhiteningmatrix, biglayers)

    % prepare the input data for the network (subtract means and whiten):
    data = grandstackable([1:end-2]) - bigmeans';
    data = (bigwhiteningmatrix*data')';
    discountedhatbest = grandstackable(:,end-1);
    discountedobs = grandstackable(:,end);
    
    % forwards pass through the network:
    layers = biglayers;
    numcases = 1;
    layers{1}.hidsums = data*layers{1}.vishid +	repmat(layers{1}.hidbiases,numcases,1);
    layers{1}.hidacts = 1./(1 + exp(-layers{1}.hidsums));
    numlayers = size(layers,2);
    for i = 2:numlayers,
        layers{i}.hidsums = layers{i-1}.hidacts*layers{i}.vishid + repmat(layers{i}.hidbiases,numcases,1);
        layers{i}.hidacts = 1./(1 + exp(-layers{i}.hidsums));
    end;

    % use the state of the neuron in the final layer of the network to
    % tweek the discounted scaled observables:
    preguess = [((2*layers{i}.hidacts)*discountedobs) discountedhatbest];    

    decision(1) = preguess(1) > preguess(2);
    decision(2) = preguess(1) <= preguess(2);
    
end % big black box
