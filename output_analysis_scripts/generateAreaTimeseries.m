%function generateAreaTimeseries
%2018-11-28
%create csv file containing area for one model and one scenario.

function generateAreaTimeseries(lakelistfile,T_list, light_list,O2_list,m1,m2,exA,y1A,exB,y1B,csvfiledir,outputdir)
%For each lake and for each day (y1A to y1B):  
%       Determine the depth (Dtemp) at which the water temperature was colder than Tc .
%       Calculate the area below Dtemp (i.e. AT = Area colder than Tc).
%       Determine the depth (Dlight) at which the light intensity was darker than  Lc.
%       Calculate the area below Dlight (i.e. AL = Area darker than Lc).
%       The niche area (Aniche) = AT – AL
%       Each Aniche < 0 is changed to 0.
%       AreaDays = (Sum of all ANiche)/10 years, giving the average of total AreaDays per year.
%       Areaday =  AreaDays/(Lake Surface Area * 365 days), giving the "proportion of the maximum potential habitat area averaged across the entire year " as calculated by Gretchen et al.
%Call the function exportAreaTimeseries to export the timeseries of NTGdays,AreaDays and Areaday for each lake into csv. 

path(path, '../outputinflow')
path(path, 'E:\output-06-08-2018')
path(path, 'E:\output-21-08-2018')

warning('off', 'all') 
startdate = [y1A, 1, 1];
i=0;
%datastart = datenum(startdate);

lakes = readLakeListCsv(lakelistfile);

nlakes = length(lakes);

timeseries_records = cell(nlakes+1, 10);
timeseries_records{1,1} = 'LakeId';
timeseries_records{1,2} = 'AreaDays';
timeseries_records{1,3} = 'Areaday';
timeseries_records{1,4} = 'SecchiMean';
timeseries_records{1,5} = 'NTGdays';
for O2_x = 1:length(O2_list)
    for T_x = 1:length(T_list)
        for Light_x = 1:length(light_list)
            fprintf(1, 'Outputting for Oxygen %d Temperature %d Light %d\n',O2_list(O2_x),T_list(T_x),light_list(Light_x));
            if isfile(sprintf('%s/fish_niche_Area_Light%.2f_T%d_O%d_%d-%d.csv',outputdir,light_list(Light_x),T_list(T_x),O2_list(O2_x),y1A,y1B))
                 continue
            else
                for lakenum = 1:nlakes
               
                    ebhex = lakes(lakenum).ebhex;
                    outputdir2 = getOutputPathFromEbHex(csvfiledir, ebhex);
                    d4 = sprintf('EUR-11_%s_%s-%s_%s_%d0101-%d1231', m1, exA, exB, m2, y1A, y1B);
                    lakedir = strcat(outputdir2, d4, '\');
    %                 disp(lakedir)

                    %try
                        T    = csvread(strcat(lakedir, 'Tzt.csv'));
                        lambdazt = csvread(strcat(lakedir, 'lambdazt.csv'));
                        ppfd = csvread(strcat(lakedir, 'PARzt.csv'));
                        O2 = csvread(strcat(lakedir, 'O2zt.csv'));
                        T_max = T_list(T_x);
                        O2_min = O2_list(O2_x);
                        %secchi_min = SD_list(SD_x);
                        ppfd_min = light_list(Light_x);

                        surface_area = lakes(lakenum).area;
                        max_depth    = lakes(lakenum).depth;

                        %area_at_depth = @(depth)( surface_area .* ((depth-1) - max_depth).^2 ./ max_depth.^2 );
                        surface_area_at_depth = @(depth)( ( surface_area .* ((depth-1) - max_depth).^2 ./ max_depth.^2 ) .*(pi*(max_depth-(depth-1)).^2 + ( surface_area .* ((depth-1) - max_depth).^2 ./ max_depth.^2 ) )).^(1/2);


                        %T_grad = diff(T')';
                        %[~, maxgrad_depth] = max(abs(T_grad'));
                        %maxgrad_depth = maxgrad_depth';
                        tlen = size(T, 1);
                        zlen = size(T, 2);
                        secchi_calculated = zeros(tlen,1);

                        area_T_below_T_max    = zeros(tlen, 1);
                        area_light_over_light_gradient = zeros(tlen, 1);
                        area_oxy_over_O2_gradient = zeros(tlen, 1);
                        area_T_and_secchi_and_O2 = zeros(tlen, 1);
                        area_over_o2_and_light_gradiant = zeros(tlen, 1);

                        for time = 1:tlen
                            %surface_attn = Attn(time, 1);
                            for depth = 1:zlen
                                if T(time, depth) < T_max

                                    area_T_below_T_max(time) = surface_area_at_depth(depth);
                                    break
                                end

                            end
                            for depth = 1:zlen
%                                 ppatmoment = ppfd(time, depth);

                                if ppfd(time, depth) < ppfd_min

                                    area_light_over_light_gradient(time) = surface_area_at_depth(depth);
                                    break
                                end

                            end
                            for depth = 1:zlen
                                if O2(time, depth) > (O2_min/0.0010) % tranform mg/l to mg/m3 (the unit use by MyLake) 

                                    area_oxy_over_O2_gradient(time) = surface_area_at_depth(depth);
                                    break
                                end

                            end
                            if zlen <5
                                lambda = lambdazt(time,1:zlen);
                            else
                                lambda = lambdazt(time,1:5);
                            end
                            secchi_calculated(time)= 1.48/mean(lambda);

                            %if secchi_calculated(time) >= secchi_min
                            %    area_secchi_under_secchi_gradient(time) = area_at_depth(secchi_calculated(time)+1);
                            %else
                            %    area_secchi_under_secchi_gradient(time) = area_at_depth(1);
                            %end
                            if area_oxy_over_O2_gradient(time) > area_light_over_light_gradient(time)
                                area_over_o2_and_light_gradiant(time)  = area_oxy_over_O2_gradient(time);
                            else
                                area_over_o2_and_light_gradiant(time)  = area_light_over_light_gradient(time);
                            end
                            area_T_and_secchi_and_O2(time)= area_T_below_T_max(time) - area_over_o2_and_light_gradiant(time);
                            if area_T_and_secchi_and_O2(time) < 0
                                area_T_and_secchi_and_O2(time) = 0;
                            end
                        end
                        ngt = sum(area_T_and_secchi_and_O2(:)==0)/10;


                        timeseries_records{1+lakenum, 1} = lakes(lakenum).lake_id;
                        timeseries_records{1+lakenum, 2} = sum(area_T_and_secchi_and_O2)/10;
                        timeseries_records{1+lakenum, 3} = (sum(area_T_and_secchi_and_O2)/10)/(surface_area_at_depth(1)*365);
                        timeseries_records{1+lakenum, 4} = mean(secchi_calculated);
                        timeseries_records{1+lakenum, 5} = ngt;


                        i = i + 1;

                    %catch
                       % warning('O2zt contain NaN')
                    %end
                end
                if i >= 1
                    name = sprintf('%s/fish_niche_Area_Light%.1f_Temperature%.1f_Oxygen%.1f_%d-%d.csv',outputdir,ppfd_min,T_max,O2_min,y1A,y1B);
                    exportAreaTimeseries(name,timeseries_records,startdate)
                end
             end
%          fprintf(1, 'Outputting for Light %d\n',light_list(Light_x));
        end
%         fprintf(1, 'Outputting for Temperature %d\n',T_list(T_x));
    end
end
end
