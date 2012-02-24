require(lme4)
require(ggplot2)
theme_set(theme_bw())

amp = 10
pwr = 2

expfull <- floor(amp*rexp(100000, rate = pwr))
exphist <- hist(expfull, breaks=max(expfull))
expkey <- ceiling(exphist$mids)
expval <- exphist$counts+1
ggplot(data.frame(expkey,expval), aes(x=expkey, y=expval)) + 
  geom_point() + geom_smooth(method="lm",se=F) + scale_y_log10()
explm <- lm(log(expval) ~ expkey)
summary(explm)
-1/explm$coefficients["expkey"]


poisfit <- data.frame(lamb=numeric(0),repsize=numeric(0),lambfit.mu=numeric(0),lambfit.sd=numeric(0))
for(lamb in c(0.2,0.5,1,2,4,8)) {
  poisfull <- rpois(100000, lamb)
  for(repsize in c(5,10,15,20,25,50,75,100)) {
    lambfits <- c()
    for(trial in 1:1000) {
      rep <- sample(poisfull, repsize)
      poishist <- hist(rep, breaks=max(rep,1),plot=F)
      poiskey <- ceiling(poishist$mids)
      poisval <- poishist$counts+1
      if(length(poiskey)>1 && length(poisval)>1) {
        poisglm <- glm(poisval ~ poiskey, family="poisson")
        lambfit <- -1/poisglm$coefficients["poiskey"]
      } else {
        lambfit <- NA
      }
      lambfits <- c(lambfits,lambfit)
    }
    poisfit <- rbind(poisfit,c(lamb,repsize,mean(lambfits,na.rm=T),sd(lambfits,na.rm=T)))
  }
}
names(poisfit) <- c("lamb","repsize","lambfit.mu","lambfit.sd")

ggplot(poisfit, aes(x=repsize,y=lambfit.mu,color=factor(lamb))) + 
  geom_line() + 
  geom_errorbar(aes(ymin=lambfit.mu-lambfit.sd,ymax=lambfit.mu+lambfit.sd)) +
  ylim(0,10)

ggplot(poisfit, aes(x=lamb,y=lambfit.mu,color=repsize,size=lambfit.sd)) + 
  geom_point() +
  scale_y_log() +
  geom_abline(slope=1, intercept=0)

ggplot(poisfit, aes(x=lamb,y=lambfit.mu,color=repsize,size=lambfit.sd)) + 
  geom_point() +
  ylim(0,10) +
  geom_abline(slope=1, intercept=0) 

lambs <- c(0.2,0.5,1,2,4,8)
repsizes <- c(5,10,15,20,25,50,75,100)
trials <- 1:1000
pois.fit <- apply(expand.grid(lambs,repsizes,trials))
