### REVISED BY RMP
### 2019 MAR 21

# author: Robert Ladwig (ladwig.jena@gmail.com)
# date: 03/11/2019
# project: automatic calibration routine for ISIMIP-GLM project
# model version: General Lake Model (GLM) v. 3.0.0rc2 
# short description: this script optimizes Sparkling Lake by comparing field with simulated water temperatures using RMSE as fit criteria
# bathymetry, meteorology and field data should be put into /bcs/
# GLM will generate its output int /output/

# CMA-ES theory: The CMA-ES implements a stochastic variable-metric method. In the very particular case of a 
# convex-quadratic objective function the covariance matrix adapts to the inverse of the Hessian matrix, up to a scalar 
# factor and small random fluctuations. The update equations for mean and covariance matrix maximize a likelihood while 
# resembling an expectation-maximization algorithm. (https://www.rdocumentation.org/packages/adagio/versions/0.7.1/topics/CMAES)

# need: R version 3.5.1 (> 3.3.2)

rm(list = ls())

# Load packages

library(rLakeAnalyzer)
library(zoo)
library(anytime)
library(optimx)
library(adagio)
library(ncdf4)
library(nloptr)
library(Hmisc)
library(matlabr)
library(xlsx)
library(tidyverse)
library(lubridate)
library(Metrics)
library(akima)
library(scales)



# Set working diretory
path <- "G:/My Drive/rachel-PC/Miami-OH/Sentinel North - Laval/MyLake_public/v12/Giles_application"
setwd(path)

# method.selec <- menu(c("Nelder-Mead", "CMA-ES"), title="Which method?")
method.selec <- 1

method.avail <- switch(method.selec, "Nelder-Mead", "CMA-ES")
print(paste('Script runs',method.avail))


# bring in observed water temperature data, clean up the data, and subset a bit as needed

setwd("G:/My Drive/rachel-PC/Miami-OH/Sentinel North - Laval/MyLake_public/v12/Giles_application")
obs.temp=read.csv("GILEStemp_all.csv",header=T)
colnames(obs.temp)=c("Date","Time","Depth","Obs_Temp","Instrument")
obs.temp.long=obs.temp %>%
  select(Date,Depth,Obs_Temp,Instrument) %>%
  mutate(Date=ymd(Date),
         Depth=Depth/-100) %>%
  filter(Date>=ymd("2017-08-11"),
         Instrument=="miniDOT",
         !Depth==0.5) %>%
  arrange(Date,Depth)


# main function

mylakeFUN <- function(p,row.locs){
  
  p <- wrapper_scales(p, lb, ub)
  
  
  ## read in parameter file, so R will automatically replace with parameter values based on optimization
  workbook <- loadWorkbook("GILES_para_v12-autoCalib.xls")
  para_file <- getSheets(workbook)[[1]]
  
  ## select parameters to optimize here:
  IscV <- p[1]
  Kz_ak_ice <- p[2]
  IscT <- p[3]
  albedo_melt_ice <- p[4]
  albedo_melt_snow <- p[5]
  C_shelter <- p[6]
  swa_b0 <- p[7]
  swa_b1 <- p[8]
 ## Kz_ak <- p[9]
  
  
  ## locations of parameter values in the spreadsheet
  addDataFrame(IscV,sheet=para_file,startRow=row.locs[1],startColumn=2,col.names=FALSE,row.names=FALSE)
  addDataFrame(Kz_ak_ice,sheet=para_file,startRow=row.locs[2],startColumn=2,col.names=FALSE,row.names=FALSE)
  addDataFrame(IscT,sheet=para_file,startRow=row.locs[3],startColumn=2,col.names=FALSE,row.names=FALSE)
  addDataFrame(albedo_melt_ice,sheet=para_file,startRow=row.locs[4],startColumn=2,col.names=FALSE,row.names=FALSE)
  addDataFrame(albedo_melt_snow,sheet=para_file,startRow=row.locs[5],startColumn=2,col.names=FALSE,row.names=FALSE)
  addDataFrame(C_shelter,sheet=para_file,startRow=row.locs[6],startColumn=2,col.names=FALSE,row.names=FALSE)
  addDataFrame(swa_b0,sheet=para_file,startRow=row.locs[7],startColumn=2,col.names=FALSE,row.names=FALSE)
  addDataFrame(swa_b1,sheet=para_file,startRow=row.locs[8],startColumn=2,col.names=FALSE,row.names=FALSE)
 ## addDataFrame(Kz_ak,sheet=para_file,startRow=row.locs[9],startColumn=2,col.names=FALSE,row.names=FALSE)
  
  saveWorkbook(workbook,"GILES_para_v12-autoCalib.xls")
  
  
  # automatically run the MyLake Matlab script via R
  setwd("G:/My Drive/rachel-PC/Miami-OH/Sentinel North - Laval/MyLake_public/v12/Giles_application")
  run_matlab_script(fname="RMC_modelGILES_v12_rmp_autoCalib.m")
  
  
  ## read in the model output, and clean up/extract the water temperature data
  file.name="ModelledTemp-Giles_2019Mar21-autoCalib.csv"
  
  mod.temp=as.data.frame(t(read.csv(file.name,header=F)))
  colnames(mod.temp)=seq(0.5,24.5,by=1)
  mod.temp.all=mod.temp %>%
    mutate(Date=seq.Date(as.Date("2016-05-17"),as.Date("2018-12-31"),by=1),
           '4'=rowMeans(cbind(`3.5`,`4.5`)),
           '6'=rowMeans(cbind(`5.5`,`6.5`)),
           '8'=rowMeans(cbind(`7.5`,`8.5`)),
           '12'=rowMeans(cbind(`11.5`,`12.5`)),
           '14'=rowMeans(cbind(`13.5`,`14.5`)),
           '16'=rowMeans(cbind(`15.5`,`16.5`)),
           '18'=rowMeans(cbind(`17.5`,`18.5`)),
           '20'=rowMeans(cbind(`19.5`,`20.5`)),
           '22'=rowMeans(cbind(`21.5`,`22.5`))) %>%
    gather(key="Depth",value="Mod_Temp",`4`:`22`) %>%
    select(Date,Depth,Mod_Temp) %>%
    mutate(Depth=as.numeric(Depth)) %>%
    arrange(Date,Depth)
  
  
  ## compare model vs. observed water temperature and return RMSE value for set of parameter values
  diag.overall=mod.temp.all %>%
    full_join(obs.temp.long) %>%
    arrange(Date,Depth) %>%
    filter(!is.na(Mod_Temp),
           !is.na(Obs_Temp)) %>%
    summarize(RMSE=rmse(actual=Obs_Temp,predicted=Mod_Temp))
  
  print(paste("Water Temp. RMSE =",signif(diag.overall[1,1],3),"°C"))
  return(diag.overall[1,1])
}    

# some theory from http://cma.gforge.inria.fr/cmaes_sourcecode_page.html: The specific formulation of a (real) optimization problem has a tremendous impact 
#on the optimization performance. In particular, a reasonable parameter encoding is essential. All parameters should be rescaled such that they have presumably 
#similar sensitivity (this makes the identity as initial covariance matrix the right choice).
wrapper_scales <- function(x, lb, ub){
  y <-  lb+(ub-lb)/(10)*(x)
  return(y)
}



# ## PARAMETER ORDER:
par.names.order=c("IscV",
                  "Kz_ak_ice",
                  "IscT",
                  "albedo_melt_ice",
                  "albedlo_melt_snow",
                  "C_shelter",
                  "swa_b0",
                  "swa_b1")



# constraints for all parameters (lb = lower bound, ub = upper bound)
lb <- c(0, 0.0001, 0, 0.1, 0.5, 0, 0.01, 0.01)
ub <- c(5, 0.016, 10, 0.7, 0.9, 1, 5, 5)

# initial guesses
values.optim <- c(2, 0.000898, 3, 0.3, 0.77, 0.1, 1, 0.5)

# row location in para_file for each p
row.locs=c(18, 5, 19, 10, 11, 7, 26, 27)


## run parameter optimization via Nelder-Mead method
if (method.selec == 1){
print("### NELDER-MEAD ###")
niter <- 5000
t1 <- Sys.time()
mylakeOPT1 <- neldermead(values.optim, mylakeFUN, lower = rep(0,length(values.optim)), 
                         upper =  rep(10,length(values.optim)), nl.info = TRUE, 
                         control=list(xtol_rel = 1e-8, maxeval = niter),
                         row.locs=row.locs)
t2 <- Sys.time()
mylakeFUN(p=mylakeOPT1$par,row.locs=row.locs)
print(paste("RMSE",mylakeOPT1$value,"°C"))
print(data.frame("Parameter"=par.names.order,
                 "OptValue"=wrapper_scales(mylakeOPT1$par,lb,ub)))
print(t2-t1)}









