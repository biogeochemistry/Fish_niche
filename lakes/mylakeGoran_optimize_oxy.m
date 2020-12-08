function ModelResult = mylakeGoran_optimize(initfile, parfile, inputfile, m_start2, m_stop2, outdir,calibration)

path(path, '../MyLake_O_simple')%change directory from mylake folder to Mylake_O_simple
path(path, '../sediments')

warning('off', 'all') 

m_start = [(m_start2-8), 1, 1]; 
ly=@(yr)~rem(yr,400)|rem(yr,100)&~rem(yr,4);
nbrmoredaysleapyears = ly((m_start2-8):(m_start2-1));
row_remove = 365*8+sum(nbrmoredaysleapyears)+1;
m_stop = [(m_stop2), 12, 31]; 

global ies80 Eevapor;
test_time = 0;
Eevapor = 0;

dt = 1.0;
try
    [In_Z,In_Az,tt,In_Tz,In_Cz,In_Sz,In_TPz,In_DOPz,In_Chlz,In_DICz,In_DOCz,In_TPz_sed,In_Chlz_sed,In_O2z,In_NO3z,In_NH4z,In_SO4z,In_HSz,In_H2Sz,In_Fe2z,In_Ca2z,In_pHz,In_CH4z,In_Fe3z,In_Al3z,In_SiO4z,In_SiO2z,In_diatomz,In_FIM,Ice0,Wt,Inflw,...
        Phys_par,Phys_par_range,Phys_par_names,Bio_par,Bio_par_range,Bio_par_names] ...
        = modelinputs_v2(m_start,m_stop, initfile, 'lake', inputfile, 'timeseries', parfile, 'lake', dt);

    % defaults a la Minnesota
    %if(isnan(Phys_par(2)))
    %  Phys_par(2) = 0.00706*(In_Az(1)/1e6)^0.56; % default diffusion coeff. parameterisation
    %end
    %
    %if(isnan(Phys_par(3)))
    %  Phys_par(3) = 8.98e-4;		%default value for diffusion coeff. in ice-covered water   
    %end
    %
    %if(isnan(Phys_par(4)))
    %  Phys_par(4) = 7e-5;			% default minimum allowed stability frequency, N2 > N0 <=> Kz < Kmax (1/s2)    		
    %end
    %
    %if(isnan(Phys_par(5)))
    %  Phys_par(5) =  1-exp(-0.3*In_Az(1)/1e6);			% default wind sheltering parameterisation		
    %end

    Depositions = 0;
    global sed_par_file;
    sed_par_file = 'sediment_parameters.txt';

    [zz,Az,Vz,tt,Qst,Kzt,Tzt,Czt,Szt,Pzt,Chlzt,PPzt,DOPzt,DOCzt,DICzt,CO2zt,O2zt,NO3zt,NH4zt,SO4zt,HSzt,H2Szt,Fe2zt,Ca2zt,pHzt,CH4zt,Fe3zt,Al3zt,SiO4zt,SiO2zt,diatomzt,O2_sat_relt,O2_sat_abst,BODzt,Qzt_sed,lambdazt,Attn_zt,PARzt,PARMaxt,P3zt_sed,P3zt_sed_sc,His,DoF,DoM,MixStat,Wt,surfaceflux,oxygenflux,CO2_eqt,~,O2_eqt,K0_O2t,CO2_ppmt,dO2Chlt,dO2BODt,dphotoDOCt,delC_org3,testi1t,testi2t,testi3t, sediments_data_basin1] = ...
        solvemodel_v2_modified(m_start,m_stop,initfile,'lake', inputfile,'timeseries', parfile,'lake',In_Z,In_Az,tt,In_Tz,In_Cz,In_Sz,In_TPz,In_DOPz,In_Chlz,In_DOCz,In_DICz,In_O2z,In_NO3z,In_NH4z,In_SO4z,In_HSz,In_H2Sz,In_Fe2z,In_Ca2z,In_pHz,In_CH4z,In_Fe3z,In_Al3z,In_SiO4z,In_SiO2z,In_diatomz,In_TPz_sed,In_Chlz_sed,In_FIM,Ice0,Wt,Inflw,Phys_par,Phys_par_range,Phys_par_names, Bio_par,Bio_par_range,Bio_par_names, Depositions);

    % NOTE: All writing of out data takes a few seconds for each file, so it should be conservative, especially in optimization runs

    
    ModelResult = compare_model_result_data2_oxy(outdir, m_start2, m_stop2,Tzt(:, row_remove:end),O2zt(:, row_remove:end),lambdazt(:, row_remove:end),calibration);
    
    f1_name = (strcat(outdir, '\Tzt.csv')); % b = binary mode, z = archived file
    dlmwrite(f1_name, Tzt(:, row_remove:end)', 'delimiter', ',', 'precision', '%.3f');

    f5_name = (strcat(outdir, '\O2zt.csv'));
    dlmwrite(f5_name, O2zt(:, row_remove:end)', 'delimiter', ',', 'precision', '%.3f');

    f7_name =(strcat(outdir, '\lambdazt.csv'));%MC uncomment to ensure creation of 2017REDOCOMPLETE
    dlmwrite(f7_name, lambdazt(:, row_remove:end)', 'delimiter', ',', 'precision', '%.3f');

    f8_name =(strcat(outdir, '\DOCzt.csv'));%MC uncomment to ensure creation of 2017REDOCOMPLETE
    dlmwrite(f8_name, DOCzt(:, row_remove:end)', 'delimiter', ',', 'precision', '%.3f');

    f6_name =(strcat(outdir, '\Qst.csv'));%MC add to ensure creation of 2017REDOCOMPLETE
    dlmwrite(f6_name, Qst(:, row_remove:end)', 'delimiter', ',', 'precision', '%.3f');

    f7_name =(strcat(outdir, '\Attn_zt.csv'));%MC 2018-05-31 add to comparaison with SDD
    dlmwrite(f7_name, Attn_zt(:, row_remove:end)', 'delimiter', ',', 'precision', '%.3f');
    
    f12_name =(strcat(outdir, '\His.csv'));%MC 2018-05-31 add to comparaison with SDD
    dlmwrite(f12_name, His(:, row_remove:end)', 'delimiter', ',', 'precision', '%.3f');

    f10_name =(strcat(outdir, '\PARzt.csv'));%MC 2018-05-31 add to comparaison with SDD
    dlmwrite(f10_name, PARzt(:, row_remove:end)', 'delimiter', ',', 'precision', '%.3f');

    f11_name =(strcat(outdir, '\PARMaxt.csv'));%MC 2018-05-31 add to comparaison with SDD
    dlmwrite(f11_name, PARMaxt(:, row_remove:end)', 'delimiter', ',', 'precision', '%.3f');
catch
      ModelResult.Dates = [];
      ModelResult.Depth = [];
      ModelResult.T_data = [];
      ModelResult.T_model = [];
end
end          