

calculThermocline = function(wtr,depths){
  #install.packages("rLakeAnalyzer")
  library(rLakeAnalyzer)

  t.d = thermo.depth(as.double(strsplit(wtr,",")[[1]]) ,as.double( strsplit(depths,',')[[1]]), seasonal=FALSE,mixed.cutoff=0.00001)
  return(t.d)
  
}



wtr = "22.51, 22.42, 22.4, 22.4, 22.4, 22.36, 22.3, 22.21, 22.11, 21.23, 16.42, 15.15, 14.24, 13.35, 10.94, 10.43, 10.36, 9.94, 9.45, 9.1, 8.91, 8.58, 8.43"
depths = "0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20"
depths = '1,2,3,4,5,6,7'
calculThermocline(wtr,depths)
