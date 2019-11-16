path(path, '../output2')
path(path, 'E:\output-06-08-2018')
path(path, 'E:\output-30-03-2019')
lakelistfile = 'D:\Fish_niche\lakes\2017SwedenList.csv';
T_list = [4,8,10,12,14,16];
light_list =[0.5, 1, 2, 4, 8 , 16, 24, 32, 48];
m1 = 'ICHEC-EC-EARTH';
m2 = 'r1i1p1_KNMI-RACMO22E_v1_day';
exA = 'historical';
exB = 'rcp45';
y1A = 2001;
y1B = 2010;
csvfiledir = 'D:\Fish_niche\output';
outputdir = 'D:\Fish_niche\output1';

%generateParamTimeseries(lakelistfile,m1,m2,exA,y1A,exB,y1B,outputdir)
%generateVolumeTimeseries(lakelistfile,m1,m2,exA,y1A,exB,y1B,outputdir)
generateAreaTimeseries(lakelistfile,T_list, light_list,m1,m2,exA,y1A,exB,y1B,csvfiledir,outputdir)