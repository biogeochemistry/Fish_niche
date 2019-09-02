%function generateVolumeTimeseries
%2018-08-03
%create csv file containing water volume for one model and one scenario.

function generateParamTimeseries(lakelistfile,m1,m2,exA,y1A,exB,y1B,outputdir)
path(path, '../output')

warning('off', 'all') 
y2B = y1B+4;
startdate = [y1A, 1, 1];
i=0;
%datastart = datenum(startdate);

lakes = readLakeListCsv(lakelistfile);


nlakes = length(lakes);

timeseries_records = cell(nlakes+1, 8);
%'Hi','Hs','Hsi','Tice','Tair','rho_snow','IceIndicator','test'
timeseries_records{1,1} = 'lakeid';
timeseries_records{1,2} = 'Hi';
timeseries_records{1,3} = 'Hs';
timeseries_records{1,4} = 'Hsi';
timeseries_records{1,5} = 'Tice';
timeseries_records{1,6} = 'Tair';
timeseries_records{1,7} = 'rho_snow';
timeseries_records{1,8} = 'IceIndicator';
timeseries_records{1,9} = 'test';


for lakenum = 1:nlakes
    
    ebhex = lakes(lakenum).ebhex;
    outputdir2 = getOutputPathFromEbHex(outputdir, ebhex);
    d4 = sprintf('EUR-11_%s_%s-%s_%s_%d0101-%d1231', m1, exA, exB, m2, y1A, y2B);
	
    lakedir = strcat(outputdir2, d4, '\');

	try
		His   = csvread(strcat(lakedir, 'His.csv'));
		disp(lakedir);
		tlen = size(His, 1);
		%zlen = size(His, 2);

		Hi = zeros(tlen, 1);
		Hs = zeros(tlen, 1);
		Hsi  = zeros(tlen, 1);
		Tice  = zeros(tlen, 1);
		Tair = zeros(tlen, 1);
		rho_snow = zeros(tlen, 1);
		IceIndicator  = zeros(tlen, 1);
		test = zeros(tlen, 1);

		for time = 1:tlen
			Hi(time) = His(time, 1);
			Hs(time) = His(time, 2);
			Hsi(time) = His(time, 3);
			Tice(time) = His(time, 4);
			Tair(time) = His(time, 5);
			rho_snow(time) = His(time, 6);
			IceIndicator(time) = His(time, 7);
			test(time) = His(time, 8);
			
		end
		timeseries_records{1+lakenum, 1} = lakes(lakenum).lake_id;
		timeseries_records{1+lakenum, 2} = Hi;
		timeseries_records{1+lakenum, 3} = Hs;
		timeseries_records{1+lakenum, 4} = Hsi;
		timeseries_records{1+lakenum, 5} = Tice;
		timeseries_records{1+lakenum, 6} = Tair;
		timeseries_records{1+lakenum, 7} = rho_snow;
		timeseries_records{1+lakenum, 8} = IceIndicator;
		timeseries_records{1+lakenum, 9} = test;
		
		i = i + 1;
    catch
		warning('His contain NaN')
		
	end
	
end
%exportParamTimeseries(sprintf('%s/fish_niche_export_His_%s.csv',outputdir,d4),timeseries_records,startdate)
%end
if i >= 1
	exportParamTimeseries(sprintf('%s/fish_niche_export_His_%s.csv',outputdir,d4),timeseries_records,startdate)
	end
end



