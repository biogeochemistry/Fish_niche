library(plotly)
library(reshape2)

#load data
data <- read.csv(file.choose())
my_df <- data[data[,8]<35,]

my_df$Lnsod
colnames(my_df)[17] <- 'LnSOD'
colnames(my_df)[5] <- 'MAX'
colnames(my_df)[18] <- 'LnArea'
SOD_lm <- lm(LnSOD ~ MAX + LnArea,data = my_df)
summary(SOD_lm)
#Graph Resolution (more important for more complex shapes)
graph_reso <- 0.05

#Setup Axis
axis_x <- seq(min(my_df$MAX), max(my_df$MAX), by = graph_reso)
axis_y <- seq(min(my_df$LnArea), max(my_df$LnArea), by = graph_reso)

#Sample points
SOD_lm_surface <- expand.grid(MAX = axis_x,LnArea = axis_y,KEEP.OUT.ATTRS = F)
SOD_lm_surface$LnSOD <- predict.lm(SOD_lm, newdata = SOD_lm_surface)
SOD_lm_surface <- acast(SOD_lm_surface, LnArea ~ MAX, value.var = "LnSOD") #y ~ x

hcolors=c("black")
SOD_plot <- plot_ly(my_df, 
                     x = ~MAX, 
                     y = ~LnArea, 
                     z = ~LnSOD,
                     text = "Species", 
                     type = "scatter3d", 
                     mode = "markers",
                     marker = list(color = hcolors))

SOD_plot <- add_trace(p = SOD_plot,
                       z = SOD_lm_surface,
                       x = axis_x,
                       y = axis_y,
                       type = "surface",
                       opacity = 0.9,
                       color="black")%>%
  layout(scene = list(xaxis = list(title = 'Max (m)'),
                    yaxis = list(title = 'Ln Area (m^2)'),
                    zaxis = list(title = 'Ln k_SOD (D^-1)')),
       annotations = list(
         x = 1.13,
         y = 1.05,
         text = 'test',
         xref = 'paper',
         yref = 'paper',
         showarrow = FALSE
       ))


SOD_plot


fit4 <-lm(log(dontest$k_SOD) ~  dontest$Max + log(dontest$Area))
hist(SOD_lm$residuals, breaks = 4)
qqnorm(SOD_lm$residuals)
qqline(SOD_lm$residuals)
shapiro.test(SOD_lm$residuals)
library(car)
library(MASS)
avPlots(fit4)
summary(SOD_lm)
hist(fit4$residuals, breaks = 4)
qqnorm(fit4$residuals)
qqline(fit4$residuals)

shapiro.test(fit4$residuals)
