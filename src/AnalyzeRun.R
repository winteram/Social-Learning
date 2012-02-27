srcdir <- "/Users/winteram/Documents/Research/Social-Learning/src/"
datadir <- "/Users/winteram/Documents/Research/Social-Learning/data/"
figdir <- "/Users/winteram/Documents/Research/Social-Learning/fig/"
setwd(srcdir)

require(ggplot2)
theme_set(theme_bw())

runname <- "SLS_run_120227_1640.csv"
rundate <- strsplit(runname, '_')[[1]][3]
runtime <- strsplit(strsplit(runname, '_')[[1]][4],'\\.')[[1]][1]
SLS <- read.csv(paste(datadir,runname,sep=""))

ggplot(SLS, aes(x=generation,y=nAgents,color=strategy)) + geom_line()
ggsave(paste(figdir,"nAgents",rundate,runtime,".pdf",sep=""),width=5,height=4)

ggplot(subset(SLS,generation>0), aes(x=generation,y=nObserve/nAgents,color=strategy)) + 
  geom_line(alpha=0.5) +
  stat_smooth(se=FALSE)
ggsave(paste(figdir,"nObserve",rundate,runtime,".pdf",sep=""),width=5,height=4)

ggplot(subset(SLS,generation>0), aes(x=generation,y=nExploit/nAgents,color=strategy)) + 
  geom_line(alpha=0.5) +
  stat_smooth(se=FALSE)
ggsave(paste(figdir,"nExploit",rundate,runtime,".pdf",sep=""),width=5,height=4)

ggplot(subset(SLS,generation>0), aes(x=generation,y=totalPayoffs,color=strategy)) + 
  geom_line(alpha=0.5) +
  stat_smooth(se=FALSE)
ggsave(paste(figdir,"Payoffs",rundate,runtime,".pdf",sep=""),width=5,height=4)

ggplot(subset(SLS,generation>0), aes(x=generation,y=avgLifespan,color=strategy)) + 
  geom_line(alpha=0.5) +
  stat_smooth(se=FALSE)
ggsave(paste(figdir,"avgLifespan",rundate,runtime,".pdf",sep=""),width=5,height=4)

ggplot(subset(SLS,generation>0), aes(x=generation,y=maxLifespan,color=strategy)) + 
  geom_line(alpha=0.5) +
  stat_smooth(se=FALSE)
ggsave(paste(figdir,"maxLifespan",rundate,runtime,".pdf",sep=""),width=5,height=4)
