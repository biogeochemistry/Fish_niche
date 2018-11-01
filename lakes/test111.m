lakelistfile = 'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\lakes\2017SwedenList.csv';
m1= 'ICHEC-EC-EARTH';
m2='r1i1p1_KNMI-RACMO22E_v1_day';
exA='historical';
y1A =1971;
exB='historical';
y1B=1980;
outputdir='G:\output-06-08-2018';
generateVolumeTimeseries(lakelistfile,m1,m2,exA,y1A,exB,y1B,outputdir)