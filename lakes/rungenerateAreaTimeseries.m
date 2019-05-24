path(path, '../output')
path(path, 'E:\output-06-08-2018')
path(path, 'E:\output-30-03-2019')

lakelistfile = '2017SwedenList111.csv';
T_list = [2,4,6,8,10,12];
SD_list =[3.25,3.5,3.75];
m1 = 'ICHEC-EC-EARTH';
m2 = 'r3i1p1_DMI-HIRHAM5_v1_day';
exA = 'historical';
exB = 'rcp45';
y1A = 2001;
y1B = 2010;
csvfiledir = 'E:\output-05-23-2019';
outputdir = 'E:\output-05-23-2019';

%generateParamTimeseries(lakelistfile,m1,m2,exA,y1A,exB,y1B,outputdir)
generateVolumeTimeseries(lakelistfile,m1,m2,exA,y1A,exB,y1B,outputdir)
%generateAreaTimeseries(lakelistfile,T_list, SD_list,m1,m2,exA,y1A,exB,y1B,csvfiledir,outputdir)