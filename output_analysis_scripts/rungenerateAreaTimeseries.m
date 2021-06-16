path(path,'F:\output')
% path(path, 'E:\output-06-08-2018')
% path(path, 'E:\output-30-03-2019')
lakelistfile = 'C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList_only_validation_12lakes.csv';

% outputdir = 'F:\output\';
list = 1:15;
T_list = [1,list(rem(list,2)==0),15];
% T_list = [15];
light_list =[0.5,1,2,4, 8, 12, 16, 20, 25,30,40,50, 60, 70, 80, 90, 100];
% light_list = [100];
oxy_list = [3.5,3,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5];
% oxy_list = [3];
m1 = 'ICHEC-EC-EARTH';
m2 = 'r3i1p1_DMI-HIRHAM5_v1_day';
exA = 'historical';
exB = 'rcp45';
y1A = 2001;
y1B = 2010;
csvfiledir = 'F:\output';
outputdir = 'F:\output\T_L_O_matrices';

%generateParamTimeseries(lakelistfile,m1,m2,exA,y1A,exB,y1B,outputdir)
%generateVolumeTimeseries(lakelistfile,m1,m2,exA,y1A,exB,y1B,outputdir)
%generateVolumeTimeseries('C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList.csv','ICHEC-EC-EARTH','r3i1p1_DMI-HIRHAM5_v1_day','historical',1971,'historical',1980,'F:\output');
generateAreaTimeseries(lakelistfile,T_list, light_list,oxy_list,m1,m2,exA,y1A,exB,y1B,csvfiledir,outputdir)