function rungenerateAreaTimeseries()

path(path,'F:\output')
% path(path, 'E:\output-06-08-2018')
% path(path, 'E:\output-30-03-2019')
lakelistfile = 'lakes\2017SwedenList.csv';

% outputdir = 'F:\output\';
list = 1:15;

T_list = [0:25];
T_list = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25];
% T_list = [15];
%light_list =[0.5,1,2,4, 8, 12, 16, 20, 25,30,40,50, 60, 70, 80, 90, 100];
light_list =[0,0.1,0.2,0.5,5,10, 15, 25, 35, 50, 65, 75, 85, 100];
% light_list = [0,0.1,0.2,5,10];
%oxy_list = [0,1.5,2.5,3.5,3,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10];
oxy_list = [0, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10];
% oxy_list = [3];
m1 = 'ICHEC-EC-EARTH';
m2 = 'DMI';
exA = 'historical';
exB = 'rcp45';
y1A = 2001;
y1B = 2010;
csvfiledir = 'lakes\stepwise_regression_result_all_model_and_scenario';
outputdir = 'lakes\T_O_L_matrices_for_surface_area_small';

%generateParamTimeseries(lakelistfile,m1,m2,exA,y1A,exB,y1B,outputdir)
%generateVolumeTimeseries(lakelistfile,'CNR','rcp85',2091,2100,'Postproc\stepwise_regression_result_all_model_and_scenario','Postproc\Habitat_Volume_by_model_and_scenario')
generateAreaTimeseries(lakelistfile,T_list, light_list,oxy_list,m2,exA,y1A,y1B,csvfiledir,outputdir)
end