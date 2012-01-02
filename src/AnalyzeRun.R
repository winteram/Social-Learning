setwd("/Users/winteram/Documents/Research/Social-Learning/data/")

require(ggplot2)
theme_set(theme_bw())

SLS <- read.csv("SLS_run_120101_2358.csv")

ggplot(SLS, aes(x=generation,y=nAgents,color=strategy)) + geom_line()

ggplot(subset(SLS,generation>0), aes(x=generation,y=nObserve,color=strategy)) + geom_line()

ggplot(subset(SLS,generation>0), aes(x=generation,y=nExploit,color=strategy)) + geom_line()

ggplot(subset(SLS,generation>0), aes(x=generation,y=totalPayoffs,color=strategy)) + geom_line()

ggplot(subset(SLS,generation>0), aes(x=generation,y=avgLifespan,color=strategy)) + geom_line()

ggplot(subset(SLS,generation>0), aes(x=generation,y=maxLifespan,color=strategy)) + geom_line()

