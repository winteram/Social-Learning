function [S Sq results ast] = tournament(pc, sigma, copy_error, nObserve, expected_lifespan, mutation_rate, full_storage, vis, runIn, varargin)
%function [S Sq results ast] = tournament(pc, sigma, copy_error, nObserve, expected_lifespan, mutation_rate, full_storage, vis, runIn, varargin{strategies})
%
% The simulation engine for running the tournament simulation described in
% the Science paper Rendell et al (2010) "Why copy others? Insights from
% the social learning strategies tournament"
%
% Strategies are expected to be found in the directory T_ROOT/strategies
% where T_ROOT is the directory containg this function.
% Strategies can be passed as a cell array of strings containing the
% strategy function filenames (e.g. {'myStrategy1.m','myStrategy2.m'}),
% otherwise the simulation will run with all functions found in
% T_ROOT/strategies. In both cases, the simulated population will initially
% contain only the first strategy listed.
%
% Input parameters are all tournament parameters excepting:
% "full_storage" is a 1/0 flag to record every detail of the simulation -
% (this results in BIG data files) - suggested default 0
%
% "vis" can be set to -1,0,1,2 depending on the level of graphics needed. The
% detailed graphics code (i.e. vis>0) is not well tested, and may not be robust. 
% Suggested setting is -1 to simply show a progress ticker in the command window
%
% "runIn" is a 1/0 flag which determines whether the initial population is
% given a 100 iteration run-in period to establish a behavioural repertoire
% before other strategies are introduced. This was always set to 1 for the
% tournament.
%
% Outputs are:
% S = the strategy makeup of the population at the simulation end
% Sq = the average frequency of each strategy in the last quarter of the simulation
% results = a structure containing some labelled information about the simulation
% ast = average strategy computation times, used to keep strategies within
% the tournament rules
%
% For more details see the paper and electronic supplements thereof.
%
% Contact: Luke Rendell <ler4@st-andrews.ac.uk>

warning off MATLAB:divideByZero
randHistory = 0; % Flag to record random generater state for testing strategy code

%_________________________________________________________________________________________________________parameters
n = 100; %the population size
n_acts = 100; %the number of possible acts

ngen = 10000; %the number of generations to run
if runIn; ngen=ngen+100;end %if one strategy is allowed to stabilise then we add 100 generations to the run...
dispRawPayoffs = 1; %if off, only fitness plotted, otherwise raw payoffs

%_________________________________________________________________________________________________________initialise
S = ones(n,1); %S will hold the current strategy list
E = round((exprnd(1,1,n_acts).^2)*2); %initialise environment

if numel(varargin)>0
    addpath([pwd filesep 'strategies']);
    strategies = varargin{1}; %if strategies provided, use them
else
    addpath([pwd filesep 'strategies']);
    strategies = dir([pwd filesep 'strategies' filesep '*.m']); %look for strategies present in the strategies directory
    strategies = {strategies.name}; %store them as names
end

n_strat = length(strategies); %count them
strategy_fns = {};
for st = 1:n_strat;
    strategies{st} = strrep(strategies{st},'.m',''); %strip off the .m
    strategy_fns{st} = eval(['@' strategies{st}]); %create a function handle and store it
end

reps = zeros(2,n_acts,n); %reps will hold the current repertoires of each individual (maximum n_acts)
history = zeros(4,max([1000 200*nObserve]),n,'int16'); %history will hold the histories for each individual

if full_storage
    if randHistory %if random state history needed...
        rand('seed',123); %set the generator state, switch to 'seed' for space 
        allHistoryRand = ones(ngen*(1/expected_lifespan)*n,max([1000 200*nObserve])); %random number generator state history for all individuals that ever lived
        historyRand = zeros(n,max([1000 200*nObserve])); %historyRand will hold the generator state histories for each individual
    end
    allHistoryRound = ones(200+ngen*(1/expected_lifespan)*n,max([1000 200*nObserve]),'int16'); %move history of all individuals that ever lived
    allHistoryMove = ones(200+ngen*(1/expected_lifespan)*n,max([1000 200*nObserve]),'int16'); %move history of all individuals that ever lived
    allHistoryAct = ones(200+ngen*(1/expected_lifespan)*n,max([1000 200*nObserve]),'int16'); %move history of all individuals that ever lived
    allHistoryPayoff = ones(200+ngen*(1/expected_lifespan)*n,max([1000 200*nObserve]),'int16'); %move history of all individuals that ever lived
    stratHistory = zeros(1,200+ngen*(1/expected_lifespan)*n,'int16'); %this will identify the strategies of all individuals that ever lived
end

meanLifePay = zeros(n,1);%current mean lifetime payoffs
cumPay = zeros(n,1); %current cumulative payoffs for each individual
roundsAlive = zeros(n,1); %how many rounds each individual has been alive
thisMove = zeros(n,3); %the last move made by each individual

stratFreqHist = zeros(n_strat,ngen); %will keep record of strategy frequencies
stratMoveHist = zeros(n_strat+1,3,ngen,'int16'); %will keep record of move history broken down by strategy
stratFreqHist(:,1) = sum(repmat([1:n_strat],n,1)==repmat(S,1,n_strat))'./n;

meanStratPayHist = zeros(n_strat,ngen); %keep record of mean strategy lifetime payoff
meanPopPayHist = zeros(1,ngen); %will keep record of mean population lifetime payoff
meanStratFitnessHist = zeros(n_strat,ngen); %will keep record of mean strategy fitness

lifespans = zeros(1,ngen*(1/expected_lifespan)*n,'int16');%record of lifespans
totalDeaths = 0; %find the number of deaths

st = zeros(n_strat,1);%record of time taken by strategies - each element will be running total time for that strategy
stc = zeros(n_strat,1); %will be a count of each time strategy is used

%Initialise screen
switch vis
    case {1 2}
        figh = figure('Units','normalized',...
            'Position',[0.2,0.05,0.6,0.9],...
            'MenuBar','none',...
            'NumberTitle','off',...
            'Toolbar','none',...
            'Name','Tournament simulation engine');
        warning off MATLAB:pie:NonPositiveData
        P_ax = axes('Units','normalized','Position',[0.05,0.15,0.9,0.4],'box','on');
        set(gca,'Ylim',[0 1],'Xlim',[1 51],'Ytick',[0 1],'Xtick',[1 51],'Xticklabel',[0 50]);
        F_ax = axes('Units','normalized','Position',[0.4,0.6,0.55,0.35],'box','on','NextPlot','replacechildren');
        set(gca,'Xlim',[0 n_strat+1],'Ytick',[],'Xtick',[1:n_strat], 'Xticklabel',strategies);
        if vis>1
            E_ax = axes('Units','normalized','Position',[0.05,0.05,0.9,0.025],'box','on');
            set(gca,'Ytick',[],'Xtick',[]);
            M_ax = axes('Units','normalized','Position',[0.05,0.8,0.3,0.15],'box','on');
            set(gca,'Ytick',[],'Xtick',[]);
            L_ax = axes('Units','normalized','Position',[0.05,0.6,0.3,0.15],'box','on');
            set(gca,'Ytick',[],'Xtick',[]);
            colormap(jet(100));
        end
        drawnow;
    case 0
        figh = figure('Units','normalized',...
            'Position',[0.25,0.05,0.5,0.9],...
            'MenuBar','none',...
            'NumberTitle','off',...
            'Toolbar','none',...
            'Name',['Tournament simulation engine']);% ' num2str(nruns*(stat-1)+thisrun) ' of ' num2str(ntot)],'Tag','progwin');
        stratlist = {};
        stratlist(([1:n_strat]*2)-1) = strategies;
        stratlist([1:n_strat]*2) = {' '};
        stratstr = {};
        stratstr(([1:n_strat]*2)-1)= repmat({'0'},1,n_strat);
        stratstr([1:n_strat]*2) = {' '};
        paystr = {};
        paystr(([1:n_strat]*2)-1)= repmat({'0'},1,n_strat);
        paystr([1:n_strat]*2) = {' '};
        gh = uicontrol('Style','text','units','normalized','position', [0.4 0.9 0.2 0.05],...
            'string','Generations:0','BackgroundColor',get(gcf,'Color'),'HorizontalAlignment','left','Fontsize',12);
        uicontrol('Style','text','units','normalized','position', [0.05 0.8 0.4 0.05],...
            'string','Strategy:','BackgroundColor',get(gcf,'Color'),'Fontsize',12);
        uicontrol('Style','text','units','normalized','position', [0.45 0.8 0.15 0.05],...
            'string','Frequency:','HorizontalAlignment','right','BackgroundColor',get(gcf,'Color'),'Fontsize',12);
        uicontrol('Style','text','units','normalized','position', [0.01 0.2 0.4 0.6],...
            'HorizontalAlignment','right','string',stratlist,'BackgroundColor',get(gcf,'Color'),'Fontsize',12);
        psh = uicontrol('Style','text','units','normalized','position', [0.45 0.2 0.1 0.6],...
            'HorizontalAlignment','right','string',stratstr,'BackgroundColor',get(gcf,'Color'),'Fontsize',12);
        switch dispRawPayoffs
            case 0
                uicontrol('Style','text','units','normalized','position', [0.65 0.8 0.15 0.05],...
                    'string','Mean fitness:','HorizontalAlignment','right','BackgroundColor',get(gcf,'Color'),'Fontsize',12);
            case 1
                uicontrol('Style','text','units','normalized','position', [0.65 0.8 0.15 0.05],...
                    'string','Mean payoffs:','HorizontalAlignment','right','BackgroundColor',get(gcf,'Color'),'Fontsize',12);
        end
        pfh = uicontrol('Style','text','units','normalized','position', [0.65 0.2 0.1 0.6],...
            'string',stratstr,'HorizontalAlignment','right','BackgroundColor',get(gcf,'Color'),'Fontsize',12);
        uicontrol('Style','text','units','normalized','position', [0.3 0.1 0.4 0.05],...
            'string','nDeaths : Lifespans (min/median/mean/max):','BackgroundColor',get(gcf,'Color'),'Fontsize',12);
        lh = uicontrol('Style','text','units','normalized','position', [0.3 0.05 0.4 0.05],'string','0 : 0 / 0 / 0 / 0','BackgroundColor',get(gcf,'Color'),'Fontsize',12);
        drawnow;
end

%____________________________________________________________________________________________________________________________________________run

for g = 1:ngen; %loop through the generations
    lastMove = thisMove; %update the record of the last move to keep data for this round's observers

    %________________________________________________________________________________________________________________MOVES
    for i=1:n; %loop through each individual
        
        currentRep = reps(:,reps(1,:,i)>0,i); %get this individuals current behavioural repertoire
        Si = S(i); %get the individual's strategy
        behavList = sort(currentRep(1,:)); %get a list of the behaviours to check that strategy doesn't alter this
        hS = find(history(1,:,i)>0,1,'last'); %find out the size of it's current history
        rA = roundsAlive(i); %how many roundsAlive
        
        if randHistory
            historyRand(i,find(historyRand(i,:)==0,1,'first')) = rand('seed'); % record random number generator state before each move made
        end
        
        tic; %start the clock
        [move currentRep] = strategy_fns{Si}(rA,currentRep,history(:,1:hS,i)); %invite the individual to move
        st(Si) = st(Si) + toc; %record the time taken by the strategy
        stc(Si) = stc(Si)+1; %add one to the count of how often stratey is used
        
        if isempty(hS); hS=0; end; %this converts empty hS to zero to give valid indexes for history later
        
        if size(currentRep,2)~=numel(behavList) %check if behaviour repertoire has changed size
            uhoh = 1;
        elseif any(sort(currentRep(1,:))~=behavList) %if same size, check that behaviour list has not changed
            uhoh = 1;
        else
            uhoh = 0; %otherwise it's OK!
        end

        if uhoh %bail if repertoire illegally modified
                    disp('***************ILLEGAL REPERTOIRE MODIFICATION*********************');
                    disp(['Player ' num2str(i) ', strategy ' strategies{S(i)} ' played illegal move: ' num2str(move)]);
                    disp('GIVEN REPERTOIRE:');
                    disp(reps(:,reps(1,:,i)>0,i));
                    disp(' ');
                    disp('RETURNED REPERTOIRE:');
                    disp(currentRep);
                    disp(' ');
                    disp('HISTORY');
                    disp(history(:,history(1,:,i)>-2,i));
                    error('tournament:IllegalRepertoireChange',['Player ' num2str(i) ', strategy ' strategies{S(i)} ' plategy ' strategies{S(i)} ' played illegal move: ' num2str(move)]);
        end

        switch move
            case -1 %INNOVATE
                unknownActs = mysetdiff([1:n_acts],currentRep(1,:)); %find acts this guys doesn't know about
                if ~isempty(unknownActs); %if don't know everything already
                    newAct = ceil(rand*length(unknownActs)); %choose a random index into unknownActs
                    act = unknownActs(newAct); %get the act at that index
                    payoff = E(act); %get that act's payoff
                    currentRep = [currentRep [act;payoff]]; %add to repertoire
                else %if you do, innovating is a stupid idea
                    act = 0;
                    payoff = 0;
                end
                
                %update history here
                history(:,hS+1,i) = [rA+1 move act payoff]; %record move in individual history
                thisMove(i,:) = [move act payoff]; %also in last move

            case 0 %OBSERVE
                lastMoveI = lastMove(i~=[1:n],:); %get the last moves of all but the focal individual
                exploiters = lastMoveI(lastMoveI(:,1)>0,:); %extract only those who exploited last round
                if ~isempty(exploiters) %if anyone did exploit last move
                    nExploit = size(exploiters,1); %the number of individuals exploiting in the last round
                    demonstrators = randperm(nExploit); %get a random index permutation into exploiters
                    if nExploit>=nObserve; %if there are sufficient observers
                        demonstrators = exploiters(demonstrators(1:nObserve),:); %then choose nObserve random exploiter(s)
                    else
                        demonstrators = [exploiters(demonstrators,:); zeros(nObserve-nExploit,3)]; %otherwise pad with zeros
                    end
                    ce = rand(nObserve,1)<=copy_error; %find copy errors
                    act = [ce.*ceil(rand(nObserve,1)*n_acts) + ~ce.*demonstrators(:,2)]; %choose an act at random if there is a copy error
                    act = act.*(demonstrators(:,1)>0); %revert any null acts to zero
                    payoff = max([round(demonstrators(:,3)+normrnd(0,sigma,nObserve,1)) zeros(nObserve,1)],[],2); %learn payoff(s) with error
                    payoff = payoff.*(demonstrators(:,1)>0); %revert any null payoffs to zero
                    for a = 1:nObserve %loop through the observed acts
                        if act(a)~=0 && any(currentRep(1,:)==act(a)) %if this act is already in the repertoire and if this was not a case of observing more than were exploiting
                            currentRep(2,find(currentRep(1,:)==act(a))) = payoff(a); %update the payoff
                        else %otherwise...
                            currentRep = [currentRep [act(a);payoff(a)]]; %update currentRep to make sure next payoff goes in right place
                        end
                    end
                else %if noone exploited then you're unlucky!
                    act = zeros(nObserve,1); %act and payoff are set to zero
                    payoff = zeros(nObserve,1);
                end

                history(:,hS+1:hS+nObserve,i) = [rA(:,ones(1,nObserve))+1; zeros(1,nObserve);act';payoff']; %record move in observe history
                thisMove(i,:) = [0 0 0]; %also in last move               
               
            otherwise %MUST BE EITHER AN ACT IN THE REPERTOIRE OR ILLEGAL
                if ~any(currentRep(1,:)==move)|| move>100 || move<-1 %check if this act is not in the repertoire or in possible move set
                    disp('***************ILLEGAL MOVE*********************');
                    disp(['Player ' num2str(i) ', strategy ' strategies{S(i)} ' played illegal move: ' num2str(move)]);
                    disp('REPERTOIRE:');
                    disp(reps(:,reps(1,:,i)>0,i));
                    disp(' ');
                    disp('HISTORY');
                    disp(history(:,history(1,:,i)>0,i));
                    %                    error('tournament:IllegalMove',['Player ' num2str(i) ', strategy ' strategies{S(i)} ' played illegal move: ' num2str(move)]);
                    act = 0;
                    payoff = 0;
                end
                act = move; %the act is the same as the move in this case
                payoff = E(act); %this is the payoff for that act
                cumPay(i) = cumPay(i) + payoff; %add this payoff to the individuals cumulative total
                currentRep(2,find(currentRep(1,:)==act)) = payoff; %update the payoff
                
                history(:,hS+1,i) = [rA+1 move act payoff]; %record move in individual history
                thisMove(i,:) = [move act payoff]; %also in last move
        end

        reps(:,1:size(currentRep,2),i) = currentRep; %update the repertoire of the individual
        
    end %next individual

    %summarise the moves and store them
    mCounts = [sum(repmat([1:n_strat],n,1)==repmat(S,1,n_strat) & repmat(thisMove(:,1),1,n_strat)==-1);...
        sum(repmat([1:n_strat],n,1)==repmat(S,1,n_strat) & repmat(thisMove(:,1),1,n_strat)==0);...
        sum(repmat([1:n_strat],n,1)==repmat(S,1,n_strat) & repmat(thisMove(:,1),1,n_strat)>0)]';
    mCounts = [mCounts; sum(mCounts,1)];
    mCounts = mCounts./repmat(sum(mCounts,2),1,3);
    stratMoveHist(:,:,g) = mCounts;

    %________________________________________________________________________________________________________________REPRODUCTION

    roundsAlive = roundsAlive+1; %increment lifespans
    meanLifePay = cumPay./roundsAlive; %recalculate the mean lifetime payoffs
    meanPopPayHist(g) = mean(meanLifePay); %store mean population lifetime payoff in history
    meanStratPay = zeros(n_strat,1); %% calculate the current average payoffs by strategy
    for ss = 1:n_strat
        meanStratPay(ss) = mean(meanLifePay(S==ss));
    end
    meanStratPayHist(1:size(meanStratPay,1),g) = meanStratPay; %store in a history matrix

    if max(meanLifePay)>0 %if payoffs have started
        relativeFitness = meanLifePay./sum(meanLifePay); %calculate relative fitness
    else
        relativeFitness = zeros(n,1); %otherwise everyone's the same
    end
    meanStratFitness = zeros(n_strat,1); %calculate mean fitnesses of each strategy
    for ss = 1:n_strat
        meanStratFitness(ss) = mean(relativeFitness(S==ss));
    end

    meanStratFitnessHist(1:size(meanStratFitness,1),g) = meanStratFitness; %store in a history matrix
    nDeaths = binornd(n,1/expected_lifespan); %find out how many die
    deaths = randperm(n); %set up random index vector
    deaths = deaths(1:nDeaths); %use ePay./sum(meanLifePay); %calculate relative fitness

    if nDeaths>0 %if there is reproduction
        cumFitness = cumsum(relativeFitness); % get cumulative sum of fitnesses
        if any(relativeFitness>0) %if all payoffs are not all zero
            newInds = sum(repmat(rand(1,nDeaths),n,1)>repmat(cumFitness,1,nDeaths))+1; %get indexes by summing logical vectors comparing cumulative fitness to random
        else %otherwise
            newInds = randperm(n); %get a random index list
            newInds = newInds(1:nDeaths); %and get a random set of individuals to replace them
        end
        %Update lists of individuals, strategies and repertoires
        newS = S(newInds); %get reproducing strategies
        if (runIn && g<101)||(n_strat>2 && (ngen-g)<(ngen/4)) %if we're not in the run in phase of a pairwise contest or the last quarter of a melee...
            muties = [];
        else
            muties = find(rand(size(newS))<=mutation_rate); %find out which ones mutate; these are now indices into newS
        end
        if ~isempty(muties) %if anyone does mutate...
            for m = 1:length(muties); %loop through mutants
                possibleS = mysetdiff([1:n_strat],newS(muties(m))); %get a set of strategies that are not equal to the one mutating from
                newS(muties(m)) = possibleS(ceil(rand*length(possibleS))); %get random element of possibleS to be new strategy
            end %next mutation
        end
        lifespans((totalDeaths+1):(totalDeaths+nDeaths)) = roundsAlive(newInds); %stores lifespan data
        if full_storage
            if (totalDeaths+nDeaths)>=200+(ngen*(1/expected_lifespan)*n); disp('WARNING!!!! allHistory matrices are now expanding - will slow me down!');end
            if size(history,2)>size(allHistoryRound,2) %check that max lifespan hasn't exceeded size of history matrices
                addition = zeros(size(allHistoryRound,1),size(history,2)-size(allHistoryRound,2)); %if it has add the required space into history matrices
                allHistoryRound = [allHistoryRound addition];
                allHistoryMove = [allHistoryMove addition];
                allHistoryAct = [allHistoryAct addition];
                allHistoryPayoff = [allHistoryPayoff addition];
                clear addition
            end
            if nDeaths==1
                allHistoryRound((totalDeaths+1):(totalDeaths+nDeaths),:) = squeeze(history(1,:,newInds)); %add the round number to allHistory
                allHistoryMove((totalDeaths+1):(totalDeaths+nDeaths),:) = squeeze(history(2,:,newInds)); % add to move history of all individuals that ever lived
                allHistoryAct((totalDeaths+1):(totalDeaths+nDeaths),:) = squeeze(history(3,:,newInds)); %act history of all individuals that ever lived
                allHistoryPayoff((totalDeaths+1):(totalDeaths+nDeaths),:) = squeeze(history(4,:,newInds)); %payoff history of all individuals that ever lived
            else                
                allHistoryRound((totalDeaths+1):(totalDeaths+nDeaths),:) = squeeze(history(1,:,newInds))'; %add the round number to allHistory
                allHistoryMove((totalDeaths+1):(totalDeaths+nDeaths),:) = squeeze(history(2,:,newInds))'; % add to move history of all individuals that ever lived
                allHistoryAct((totalDeaths+1):(totalDeaths+nDeaths),:) = squeeze(history(3,:,newInds))'; %act history of all individuals that ever lived
                allHistoryPayoff((totalDeaths+1):(totalDeaths+nDeaths),:) = squeeze(history(4,:,newInds))'; %payoff history of all individuals that ever lived
            end
            if randHistory
                allHistoryRand((totalDeaths+1):(totalDeaths+nDeaths),:) = historyRand(newInds,:); % save the generator state history
            end
            stratHistory((totalDeaths+1):(totalDeaths+nDeaths)) = S(newInds); %add strategies to history
        end
        lastS = S; %store last S
        S(deaths) = newS; %put new individuals into population
        reps(:,:,deaths) = 0; %clear the repertoires of replaced individuals
        history(:,:,deaths) = 0; %clear the histories of replaced individuals
        if randHistory
            historyRand(deaths,:) = 0; %if recording random history, clear the histories of replaced individuals
        end
        roundsAlive(deaths) = 0; %reset lifespansaths) = 0; %reset lifespans
        cumPay(deaths) = 0; %reset cumulative payoffs
        totalDeaths = totalDeaths+nDeaths; %update death count
        %disp([ num2str(numReproducing) ' reproduced']);
    end

    stratFreq = sum(repmat([1:n_strat],n,1)==repmat(S,1,n_strat))'./n; %calculate the frequencies of each strategy
    stratFreqHist(:,g) = stratFreq; %store it in a history matrix
    %________________________________________________________________________________________________________________ENVIRONMENT

    changeP = rand(1,n_acts)<=pc; %find out which payoffs change
    E(find(changeP)) = round((exprnd(1,1,sum(changeP)).^2)*2); %update payoffs

    %________________________________________________________________________________________________________________UPDATE SCREEN
    switch vis
        case {1 2} %high or very high
            axes(P_ax)
            plot(stratFreqHist(:,1:g)');
            legend(strategies, 'Location','NorthWest');
            hold on
            plot(meanStratFitnessHist(:,1:g)',':');
            hold off
            if g>50
                set(P_ax,'Xlim', [1 g],'Ylim',[0 1],'Ytick',[0 1], 'Xtick',[1 g]);
            else
                set(P_ax,'Ylim',[0 1],'Xlim',[1 51],'Ytick',[0 1],'Xtick',[1 51]);
            end
            xlabel('Rounds');


            if g>1
                axes(F_ax);
                switch dispRawPayoffs
                    case 0
                        boxplot(relativeFitness,S);
                        hold on
                        plot([1:length(unique(S))],meanStratFitness(unique(S)),'g*');
                        hold off
                        set(F_ax,'Xlim',[0 length(unique(S))+1],'YLim',[0 1],'Ytick',[0 1],'Xtick',[1:length(unique(S))], 'Xticklabel',strategies(unique(S)));
                    case 1
                        oldY = max(get(F_ax,'Ylim'));
                        boxplot(meanLifePay,S);
                        hold on
                        plot([1:length(unique(S))],meanStratPay(unique(S)),'g*');
                        set(F_ax,'NextPlot','replacechildren');
                        set(F_ax,'Xlim',[0 length(unique(S))+1],'Xtick',[1:length(unique(S))], 'Xticklabel',strategies(unique(S)));
                        set(F_ax,'Ylim',[0 max([oldY ceil(max(meanLifePay))])],'Ytick',[0 max([oldY ceil(max(meanLifePay))])]);
                end
                ylabel('');
            end

            if g>1 && vis>1
                axes(E_ax);
                exploitIm = thisMove(thisMove(:,1)>0,2);
                image(sort(exploitIm)','parent',E_ax);
                set(E_ax,'Xtick',[],'Ytick',[]);

                axes(M_ax);
                bar(mCounts,'stacked');
                oldX = get(M_ax,'Xticklabel');
                oldX(n_strat+1) = 'A';
                set(M_ax,'Xticklabel',oldX);
                legend({'I','O','E'},'location','Eastoutside');

                if any(lifespans>0)
                    axes(L_ax);
                    h = hist(lifespans(1:totalDeaths),[1:max(lifespans)]);
                    h = h./sum(h);
                    bar([1:max(lifespans)],h);
                    hold on;
                    plot([mean(lifespans(1:totalDeaths)).*ones(2,1)],get(L_ax,'Ylim'),'r');
                    plot([mean(roundsAlive).*ones(2,1)],get(L_ax,'Ylim'),'g');
                    hold off;
                end
            end
            drawnow;
            
        case 0 %low
            set(gh,'String',['Generations: ' num2str(g)]);
            for sl = 1:n_strat;
                stratstr((sl*2)-1) = {sprintf('%4.2f', stratFreq(sl))};
                switch dispRawPayoffs
                    case 0
                        paystr((sl*2)-1) = {sprintf('%4.2f', meanStratFitness(sl))};
                    case 1
                        paystr((sl*2)-1) = {sprintf('%4.2f', round(meanStratPay(sl)))};
                end
            end
            set(psh,'String',stratstr);
            set(pfh,'String',paystr);
            set(lh,'String',sprintf('%4.0f : %4.0f / %4.0f / %4.0f / %4.0f',...
                double(totalDeaths), double(min(lifespans(1:totalDeaths))),double(median(lifespans(1:totalDeaths))),double(mean(lifespans(1:totalDeaths))),double(max(lifespans(1:totalDeaths)))));
            drawnow;
        otherwise
            if mod(g,1000)==0
                fprintf(1,'|')
            elseif mod(g,200)==0
                fprintf(1,'-')
            end
    end


end %next generation

ast = st./stc;

disp(' COMPLETED');

if vis==0; close(figh);end

if vis==0 %plot summary at end
    disp('Average strategy times:');
    disp([[1:n_strat]' ast]);
    disp(['Average lifespan: ' num2str(mean(lifespans(lifespans>0)))]);
    disp(['Median lifespan: ' num2str(median(lifespans(lifespans>0)))]);
    disp(['Maximum lifespan: ' num2str(max(lifespans(lifespans>0)))]);
    figure; %plot history
    plot(stratFreqHist(:,1:g)');
    legend(strategies);
    set(gca,'Xlim', [1 g],'Ylim',[0 1],'Ytick',[0 1], 'Xtick',[1 g]);
end

Sq = mean(stratFreqHist(:,(ngen-(round(ngen/4))):ngen),2); %get mean over last 1/4 generations
results = struct('pc',pc,'mutation_rate',mutation_rate,'sigma',sigma,'copy_error',copy_error,'nObserve',nObserve,'expected_lifespan',expected_lifespan,...
    'averageStrategyTimes',ast,'reps',reps,'history',history,'S',S,'S_lastQuarter',Sq,'roundsAlive',roundsAlive,'thisMove',thisMove,...
    'stratMoveHist',stratMoveHist,'stratFreqHist',stratFreqHist,'meanStratPayHist',meanStratPayHist,'meanPopPayHist',meanPopPayHist,'meanStratFitnessHist',meanStratFitnessHist,...
    'lifespans',lifespans,'totalDeaths',totalDeaths);
results.strategies = strategies;

if full_storage
    results(1).allHistoryRound = allHistoryRound;
    clear allHistoryRound;
    results(1).allHistoryMove = allHistoryMove;
    clear allHistoryMove;
    results(1).allHistoryAct = allHistoryAct;
    clear allHistoryAct;
    results(1).allHistoryPayoff = allHistoryPayoff;
    clear allHistoryPayoff;
    results(1).stratHistory = stratHistory;
    clear stratHistory;
end

if randHistory
    results(1).historyRand = historyRand;
    results(1).allHistoryRand = allHistoryRand;
    clear allHistoryRand
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C = mysetdiff(A,B)

% if isempty(A)
%     C = [];
%     return;
if isempty(B)
    C = A;
    return; 
else % both non-empty
    bits = zeros(1, max(max(A), max(B)));
    bits(A) = 1;
    bits(B) = 0;
    C = A(logical(bits(A)));
end

