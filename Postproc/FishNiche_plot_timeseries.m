


% The date should be dynamically loaded based on choice of scenario
startdate = [2001, 1, 1];
datastart = datenum(startdate);

nlakes = size(timeseries_records, 1) - 1;
%lakerange: which lakes to plot. should be subset of 1:nlakes
lakerange = 1:nlakes;

plotncolumns = 4;
plotnrows = ceil(length(lakerange) / plotncolumns);
hfig = figure();

for lakenum = lakerange

    timeaxis = datastart:(datastart+tlen-1);
    ax = subplot(plotnrows, plotncolumns, lakenum);

    hold(ax, 'on');
    
    volume_T_below_15 = timeseries_records{1+lakenum, 6};
    volume_O2_above_3000 = timeseries_records{1+lakenum, 7};

    plot(timeaxis, volume_T_below_15, 'b');
    plot(timeaxis, volume_O2_above_3000, 'r');
    datetick('x', 'yyyy');
    xlabel(lakes(lakenum).name);
    v = axis;
    v(1) = datastart;
    v(2) = datastart+tlen-1;
    axis(v);

    hold(ax, 'off');
end