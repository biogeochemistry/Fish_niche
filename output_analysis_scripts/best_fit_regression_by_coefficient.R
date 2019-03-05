
data <- read.csv(file="ResultCalibration12Lakes.csv")


#library needed to be imported
library(leaps)
library(car)

### Data visualisation ###

# plots each coefficient with variables to validate if need log transformation #

hist(data$SWA_B1)
hist(data$k_SOD)
hist(data$k_BOD)
# 3 distributions need to be transformed
hist(log(data$SWA_B1))
hist(log(data$k_SOD))
hist(log(data$k_BOD))

# SWA_b1 is corrected, others are still not normalized (but are better). 

plot(log(SWA_B1) ~ log(Area), data=data) 
plot(log(SWA_B1) ~ log(MaxDepth), data=data)
plot(log(SWA_B1) ~log(MeanDepth), data=data)
plot(log(SWA_B1) ~ log(CatchementArea), data=data)
plot(log(SWA_B1) ~ log(RatioC.A), data=data)
plot(log(SWA_B1) ~ log(Sediment), data=data)
plot(log(SWA_B1) ~ log(Volume), data=data)

#they are all better with log transformation

plot(log(k_SOD) ~ log(Area), data=data) 
plot(log(k_SOD) ~ (MaxDepth), data=data)
plot(log(k_SOD) ~ (MeanDepth), data=data)
plot(log(k_SOD) ~ log(CatchementArea), data=data)
plot(log(k_SOD) ~ log(RatioC.A), data=data)
plot(log(k_SOD) ~ log(Sediment), data=data)
plot(log(k_SOD) ~ log(Volume), data=data)

#Only 2 variables don't need transformation

plot(log(k_BOD) ~ log(Area), data=data) 
plot(log(k_BOD) ~ (MaxDepth), data=data)
plot(log(k_BOD) ~ (MeanDepth), data=data)
plot(log(k_BOD) ~ log(CatchementArea), data=data)
plot(log(k_BOD) ~ log(RatioC.A), data=data)
plot(log(k_BOD) ~ log(Sediment), data=data)
plot(log(k_BOD) ~ log(Volume), data=data)

#Same as k_BOD


###  SWA_B1 regression ###

leaps<-regsubsets((SWA_B1) ~ log(MeanDepth) + log(MaxDepth) + log(Area) +log(CatchementArea)+ log(Sediment) + log(RatioC.A) +log(Volume), data=data,nbest=10)
summary(leaps)
# models are ordered by the selection statistic.
plot(leaps,scale="r2")
# plot statistic by subset size 
subsets(leaps, statistic="adjr2")

#BEST FIT TEST (12 Lakes)
fit <- lm(SWA_B1  ~ 1 , data=data)

fit1 <- lm(SWA_B1 ~ log(MaxDepth) , data=data)
fit2 <- lm(SWA_B1 ~ log(MaxDepth) + log(RatioC.A), data=data)
fit3 <- lm(SWA_B1 ~ log(MaxDepth) + log(Area), data=data)
fit4 <- lm(SWA_B1 ~ log(MaxDepth) + log(Volume), data=data)
fit5 <- lm(SWA_B1 ~ log(Volume)+ log(Area), data=data)

anova(fit,fit1)
anova(fit1,fit2)
anova(fit1,fit3)
anova(fit1,fit4)
anova(fit1,fit5)

hist(fit1$residuals, breaks = 4)
qqnorm(fit1$residuals)
qqline(fit1$residuals)
shapiro.test(fit1$residuals)

#Visualization
plot(SWA_B1 ~ log(MaxDepth) + log(Volume), data=data)
abline(lm(SWA_B1 ~ log(MaxDepth) + log(Volume), data=data))


#BEST FIT TEST (10 Lakes)
data_10 <- don[don[,11]>10000000,]
leaps<-regsubsets(SWA_B1 ~ MeanDepth + MaxDepth + Area + Aire_sedimentaire + Ratio.RatioC.A+Volume, data=don,nbest=10)
summary(leaps)
# models are ordered by the selection statistic.
plot(leaps,scale="r2")
# plot statistic by subset size 
library(car)
subsets(leaps, statistic="adjr2")
fit <- lm(SWA_B1  ~ 1 , data=data_10)

fit1 <- lm(SWA_B1 ~ log(MaxDepth) , data=data_10)
fit2 <- lm(SWA_B1 ~ log(MaxDepth) + log(RatioC.A), data=data_10)
fit3 <- lm(SWA_B1 ~ log(MaxDepth) + log(Area), data=data_10)
fit4 <- lm(SWA_B1 ~ log(MaxDepth) + log(Volume), data=data_10)
fit5 <- lm(SWA_B1 ~ log(Volume)+ log(Area), data=data_10)

anova(fit,fit1)
anova(fit1,fit2)
anova(fit1,fit3)
anova(fit1,fit4)
anova(fit1,fit5)


hist(fit1$residuals, breaks = 4)
qqnorm(fit1$residuals)
qqline(fit1$residuals)
shapiro.test(fit1$residuals)

#Visualization
plot(SWA_B1 ~ log(MaxDepth), data=data_10)
abline(lm(SWA_B1 ~ log(MaxDepth), data=data_10))



#I-scDOC
plot(I_scDOC ~  SWA_B1)



#K_BOD
hist(K_BOD,data=data)#not normal distributed
leaps<-regsubsets(log(K_BOD) ~ poly(MeanDepth,2) + poly(MaxDepth,2) + log(Area)
                  + log(Ratio.RatioC.A)+ log(Volume) + log(Area), data=data,nbest=10)
summary(leaps)
# models are ordered by the selection statistic.
plot(leaps,scale="r2")
# plot statistic by subset size 
library(car)
subsets(leaps, statistic="adjr2")

#BEST FIT TEST(12 Lakes)
fit <- lm(log(K_BOD)  ~ 1 , data=data)

fit1 <- lm(log(K_BOD) ~ MeanDepth , data=data)
fit2 <- lm(log(K_BOD) ~ MaxDepth, data=data)
fit3 <- lm(log(K_BOD) ~ MeanDepth + Area, data=data)
fit4 <- lm(log(K_BOD) ~ MaxDepth + Area, data=data)
fit5 <- lm(log(K_BOD) ~ MaxDepth + Volume, data=data)
fit6 <- lm(log(K_BOD) ~ I(MeanDepth^2)+ MeanDepth, data=data)
fit7 <- lm(log(K_BOD) ~ I(MeanDepth^2)+ Area + Volume, data=data)

anova(fit,fit1)
anova(fit,fit2)
anova(fit1,fit3)
anova(fit2,fit4)
anova(fit2,fit5)
anova(fit1,fit6)
anova(fit1,fit7)
BIC(fit1,fit2)

hist(fit2$residuals, breaks = 4)
qqnorm(fit2$residuals)
qqline(fit2$residuals)
shapiro.test(fit2$residuals)

#Visualization
plot(log(K_BOD) ~ MaxDepth, data=data)
abline(lm(log(K_BOD) ~ MaxDepth, data=data))


#BEST FIT TEST (10 Lakes)

leaps<-regsubsets(log(K_BOD) ~ poly(MeanDepth,2) + poly(MaxDepth,2) + log(Area)
                  + log(Ratio.RatioC.A)+ log(Volume) + log(Area), data=data_10,nbest=10)
summary(leaps)
# models are ordered by the selection statistic.
plot(leaps,scale="r2")
# plot statistic by subset size 
library(car)
subsets(leaps, statistic="adjr2")

fit <- lm(log(K_BOD)  ~ 1 , data=data_10)

fit1 <- lm(log(K_BOD) ~ MeanDepth , data=data_10)
fit2 <- lm(log(K_BOD) ~ MaxDepth, data=data_10)
fit3 <- lm(log(K_BOD) ~ MeanDepth + MaxDepth, data=data_10)
fit4 <- lm(log(K_BOD) ~ MeanDepth+ I(MaxDepth^2), data=data_10)
fit5 <- lm(log(K_BOD) ~ MeanDepth + Volume, data=data_10)
fit6 <- lm(log(K_BOD) ~ I(MeanDepth^2)+ MeanDepth, data=data_10)
fit7 <- lm(log(K_BOD) ~ MeanDepth + Area, data=data_10)

anova(fit,fit1)
anova(fit,fit2)
anova(fit1,fit3)
anova(fit2,fit3)
anova(fit1,fit4)
anova(fit2,fit4)
anova(fit1,fit5)
anova(fit1,fit6)
anova(fit1,fit7)
BIC(fit1,fit2)

hist(fit1$residuals, breaks = 4)
qqnorm(fit1$residuals)
qqline(fit1$residuals)
shapiro.test(fit1$residuals)

#Visualization
plot(log(K_BOD) ~ MeanDepth, data=data_10)
abline(lm(log(K_BOD) ~ MeanDepth, data=data_10))


#K_SOD
hist(K_SOD,data=data)

leaps<-regsubsets(log(k_SOD) ~ log(MaxDepth)+MaxDepth+MeanDepth+log(MeanDepth)+log(Ratio.RatioC.A)+log(Volume) + log(Area),
                  data=data,nbest=10)
# view results 
summary(leaps)
# plot a table of models showing variables in each model.
# models are ordered by the selection statistic.
plot(leaps,scale="adjr2")

fitsod <- lm(log(k_SOD)~1,data=data)
fitsod1 <- lm(log(k_SOD)~MaxDepth+log(area),data=data)
fitsod2 <- lm(log(k_SOD)~MaxDepth+log(area)+log(Ratio.RatioC.A),data=data)
fitsod3 <- lm(log(k_SOD)~MaxDepth + log(area)+log(MeanDepth),data=data)
anova(fitsod1,fitsod2)
anova(fitsod1,fitsod3)

data_8 <- data[data[,8]<30,]

leaps<-regsubsets(log(k_SOD) ~ log(MaxDepth)+MaxDepth+MeanDepth+log(MeanDepth)+log(RatioC.A)+log(Volume) + log(Area),
                  data=data_8,nbest=10)
# view results 
summary(leaps)
# plot a table of models showing variables in each model.
# models are ordered by the selection statistic.
plot(leaps,scale="adjr2")

fitsod1 <- lm(log(k_SOD)~MeanDepth,data=dontest)
fitsod2 <- lm(log(k_SOD)~MaxDepth,data=dontest)
fitsod3 <- lm(log(k_SOD)~MeanDepth + log(Area),data=dontest)
fitsod4 <- lm(log(k_SOD)~MaxDepth + log(Area),data=dontest)
fitsod5 <- lm(log(k_SOD)~MeanDepth + log(MaxDepth)+log(Volume),data=dontest)
anova(fitsod1,fitsod3)
anova(fitsod2,fitsod4)
anova(fitsod1,fitsod5)
anova(fitsod3,fitsod5)
BIC(fitsod3,fitsod4)
summary(fitsod3)


hist(fitsod3$residuals, breaks = 4)
qqnorm(fitsod3$residuals)
qqline(fitsod3$residuals)
shapiro.test(fitsod3$residuals)
#p non significative, donc residues suit une loi normale


