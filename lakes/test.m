%test
filename = "C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017par";
%fid = readtable(filename, 'HeaderLines', 1,'Format','%s %f %f %f %s');
     
%mylakeGoran(,2008,2014,'D:\output\NO\Langtjern\EWEMBI\historical')
%tes3t = fid(:,{'Value','Parameter'});
m_start = 1971;
m_stop= 1980;
initfile = 'C:\Users\macot620\Documents\GitHub\Fish_niche\output\fb\fb9a\fb9a8c\EUR-11_ICHEC-EC-EARTH_historical-historical_r1i1p1_KNMI-RACMO22E_v1_day_19710101-19801231\2017init';

inputfile = 'C:\Users\macot620\Documents\GitHub\Fish_niche\output\fb\fb9a\fb9a8c\EUR-11_ICHEC-EC-EARTH_historical-historical_r1i1p1_KNMI-RACMO22E_v1_day_19710101-19801231\2017input';

parfile = 'C:\Users\macot620\Documents\GitHub\Fish_niche\output\fb\fb9a\fb9a8c\EUR-11_ICHEC-EC-EARTH_historical-historical_r1i1p1_KNMI-RACMO22E_v1_day_19710101-19801231\2020par';
outdir = 'C:\Users\macot620\Documents\GitHub\Fish_niche\output\fb\fb9a\fb9a8c\EUR-11_ICHEC-EC-EARTH_historical-historical_r1i1p1_KNMI-RACMO22E_v1_day_19710101-19801231';
latitude = 57.912728;
longitude = 14.81677;


%tes1t = load_params_lake(63,13);
%a=1;
%export_params_lake(tes1t, filename)
%outdir = "D:\output\AU\Argyle\EWEMBI\historical";
%m_start2 = 1976;
%m_stop2= 2016;
%ModelResult = compare_model_result_data(outdir, m_start2, m_stop2)
%mylakeGoran('F:\output\fa\faed\faed2d\test\2020init','F:\output\fa\faed\faed2d\test\2020par','F:\output\fa\faed\faed2d\test\2020input',1999,2010,'F:\output\fa\faed\faed2d\test');
%mylakeGoran_optimize('C:\Users\macot620\Documents\GitHub\Fish_niche\output\fa\faed\faed2d\EUR-11_ICHEC-EC-EARTH_historical-historical_r3i1p1_DMI-HIRHAM5_v1_day_19710101-19801231\2020init','C:\Users\macot620\Documents\GitHub\Fish_niche\output\fa\faed\faed2d\EUR-11_ICHEC-EC-EARTH_historical-historical_r3i1p1_DMI-HIRHAM5_v1_day_19710101-19801231\2020par','C:\Users\macot620\Documents\GitHub\Fish_niche\output\fa\faed\faed2d\EUR-11_ICHEC-EC-EARTH_historical-historical_r3i1p1_DMI-HIRHAM5_v1_day_19710101-19801231\2020input',1971,1980,'C:\Users\macot620\Documents\GitHub\Fish_niche\output\fa\faed\faed2d\EUR-11_ICHEC-EC-EARTH_historical-historical_r3i1p1_DMI-HIRHAM5_v1_day_19710101-19801231',1);
%Result = MyLake_optimizer(2001,2010,'F:\output\fa\faed\faed2d\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231\2020par','F:\output\fa\faed\faed2d\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231\2020input','F:\output\fa\faed\faed2d\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231\2020init','F:\output\fa\faed\faed2d\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231',61.327758,14.273501);
%Result = MyLake_optimizer(2001,2010,'F:\output\fa\faed\faed2d\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231\2020par','F:\output\fa\faed\faed2d\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231\2020input','F:\output\fa\faed\faed2d\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231\2020init','F:\output\fa\faed\faed2d\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231',61.327758,14.273501);
%Result = MyLake_optimizer(2001,2010,'F:\output\fb\fb9a\fb9a8c\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231\2020par','F:\output\fb\fb9a\fb9a8c\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231\2020input','F:\output\fb\fb9a\fb9a8c\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231\2020init','F:\output\fb\fb9a\fb9a8c\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231',57.912728,14.816770);
%parfile = "F:\output\fa\faed\faed2d\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231\2019par";
%parfile2 = "F:\output\fa\faed\faed2d\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231\2020par";
Result = MyLake_optimizer(2001,2010,'F:\output\fb\fb9a\fb9a8c\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231\2020par','F:\output\fb\fb9a\fb9a8c\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231\2020input','F:\output\fb\fb9a\fb9a8c\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231\2020init','F:\output\fb\fb9a\fb9a8c\EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231',57.912728,14.816770);
% optimal_parameters = [0.0865, 0.436, 104, 3];
% optimal_parameterss = optimal_parameters.';
% export_params_lake(optimal_parameterss,parfile2, parfile)
% 
% test111 = Result;