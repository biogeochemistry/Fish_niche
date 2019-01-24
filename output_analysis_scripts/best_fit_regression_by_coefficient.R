
data <- read.csv (file.choose())

#SWA_B1
library(leaps)
attach(don)
leaps<-regsubsets(SWA_B1 ~ mean_depth + max.depth + Area + Aire_sedimentaire + Ratio.B.L+volume, data=don,nbest=10)
summary(leaps)
# models are ordered by the selection statistic.
plot(leaps,scale="r2")
# plot statistic by subset size 
library(car)
subsets(leaps, statistic="adjr2")

#BEST FIT TEST (12 Lakes)
fit <- lm(SWA_B1  ~ 1 , data=data)

fit1 <- lm(SWA_B1 ~ log(max.depth) , data=data)
fit2 <- lm(SWA_B1 ~ log(max.depth) + log(B.L), data=data)
fit3 <- lm(SWA_B1 ~ log(max.depth) + log(Area), data=data)
fit4 <- lm(SWA_B1 ~ log(max.depth) + log(volume), data=data)
fit5 <- lm(SWA_B1 ~ log(volume)+ log(Area), data=data)

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
plot(SWA_B1 ~ log(max.depth) + log(volume), data=data)
abline(lm(SWA_B1 ~ log(max.depth) + log(volume), data=data))


#BEST FIT TEST (10 Lakes)
data_10 <- don[don[,11]>10000000,]
leaps<-regsubsets(SWA_B1 ~ mean_depth + max.depth + Area + Aire_sedimentaire + Ratio.B.L+volume, data=don,nbest=10)
summary(leaps)
# models are ordered by the selection statistic.
plot(leaps,scale="r2")
# plot statistic by subset size 
library(car)
subsets(leaps, statistic="adjr2")
fit <- lm(SWA_B1  ~ 1 , data=data_10)

fit1 <- lm(SWA_B1 ~ log(max.depth) , data=data_10)
fit2 <- lm(SWA_B1 ~ log(max.depth) + log(B.L), data=data_10)
fit3 <- lm(SWA_B1 ~ log(max.depth) + log(Area), data=data_10)
fit4 <- lm(SWA_B1 ~ log(max.depth) + log(volume), data=data_10)
fit5 <- lm(SWA_B1 ~ log(volume)+ log(Area), data=data_10)

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
plot(SWA_B1 ~ log(max.depth), data=data_10)
abline(lm(SWA_B1 ~ log(max.depth), data=data_10))



#I-scDOC
plot(I_scDOC ~  SWA_B1)



#K_BOD
hist(K_BOD,data=data)#not normal distributed
leaps<-regsubsets(log(K_BOD) ~ poly(mean_depth,2) + poly(max.depth,2) + log(Area)
                  + log(Ratio.B.L)+ log(volume) + log(Area), data=data,nbest=10)
summary(leaps)
# models are ordered by the selection statistic.
plot(leaps,scale="r2")
# plot statistic by subset size 
library(car)
subsets(leaps, statistic="adjr2")

#BEST FIT TEST(12 Lakes)
fit <- lm(log(K_BOD)  ~ 1 , data=data)

fit1 <- lm(log(K_BOD) ~ mean_depth , data=data)
fit2 <- lm(log(K_BOD) ~ max.depth, data=data)
fit3 <- lm(log(K_BOD) ~ mean_depth + Area, data=data)
fit4 <- lm(log(K_BOD) ~ max.depth + Area, data=data)
fit5 <- lm(log(K_BOD) ~ max.depth + volume, data=data)
fit6 <- lm(log(K_BOD) ~ I(mean_depth^2)+ mean_depth, data=data)
fit7 <- lm(log(K_BOD) ~ I(mean_depth^2)+ Area + volume, data=data)

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
plot(log(K_BOD) ~ max.depth, data=data)
abline(lm(log(K_BOD) ~ max.depth, data=data))


#BEST FIT TEST (10 Lakes)

leaps<-regsubsets(log(K_BOD) ~ poly(mean_depth,2) + poly(max.depth,2) + log(Area)
                  + log(Ratio.B.L)+ log(volume) + log(Area), data=data_10,nbest=10)
summary(leaps)
# models are ordered by the selection statistic.
plot(leaps,scale="r2")
# plot statistic by subset size 
library(car)
subsets(leaps, statistic="adjr2")

fit <- lm(log(K_BOD)  ~ 1 , data=data_10)

fit1 <- lm(log(K_BOD) ~ mean_depth , data=data_10)
fit2 <- lm(log(K_BOD) ~ max.depth, data=data_10)
fit3 <- lm(log(K_BOD) ~ mean_depth + max.depth, data=data_10)
fit4 <- lm(log(K_BOD) ~ mean_depth+ I(max.depth^2), data=data_10)
fit5 <- lm(log(K_BOD) ~ mean_depth + volume, data=data_10)
fit6 <- lm(log(K_BOD) ~ I(mean_depth^2)+ mean_depth, data=data_10)
fit7 <- lm(log(K_BOD) ~ mean_depth + Area, data=data_10)

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
plot(log(K_BOD) ~ mean_depth, data=data_10)
abline(lm(log(K_BOD) ~ mean_depth, data=data_10))


#K_SOD
hist(K_SOD,data=data)

leaps<-regsubsets(log(k_SOD) ~ log(max.depth)+max.depth+mean_depth+log(mean_depth)+log(Ratio.B.L)+log(volume) + log(Area),
                  data=data,nbest=10)
# view results 
summary(leaps)
# plot a table of models showing variables in each model.
# models are ordered by the selection statistic.
plot(leaps,scale="adjr2")

fitsod <- lm(log(k_SOD)~1,data=data)
fitsod1 <- lm(log(k_SOD)~max.depth+log(area),data=data)
fitsod2 <- lm(log(k_SOD)~max.depth+log(area)+log(Ratio.B.L),data=data)
fitsod3 <- lm(log(k_SOD)~max.depth + log(area)+log(mean_depth),data=data)
anova(fitsod1,fitsod2)
anova(fitsod1,fitsod3)

data_8 <- data[data[,8]<30,]

leaps<-regsubsets(log(k_SOD) ~ log(max.depth)+max.depth+mean_depth+log(mean_depth)+log(Ratio.B.L)+log(volume) + log(Area),
                  data=data_8,nbest=10)
# view results 
summary(leaps)
# plot a table of models showing variables in each model.
# models are ordered by the selection statistic.
plot(leaps,scale="adjr2")

fitsod1 <- lm(log(k_SOD)~mean_depth,data=dontest)
fitsod2 <- lm(log(k_SOD)~max.depth,data=dontest)
fitsod3 <- lm(log(k_SOD)~mean_depth + log(Area),data=dontest)
fitsod4 <- lm(log(k_SOD)~max.depth + log(Area),data=dontest)
fitsod5 <- lm(log(k_SOD)~mean_depth + log(max.depth)+log(volume),data=dontest)
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


