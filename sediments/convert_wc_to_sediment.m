function [sediments_bc] = convert_wc_to_sediment(MyLake_concentrations, MyLake_params, sediments_params)
%convert_wc_to_sediment function convert BC values for sediments module (different Units)
    %   [Water-column units] ------>  [Sediments units]
    % for dissolved species - concentration BC
    % for solid - Neumann (flux) BC
    sediments_bc = {...
        dissolved_bc(MyLake_concentrations('O2z'), 31998.8),                           'Ox_c';
        OM1_flux(MyLake_concentrations, MyLake_params, sediments_params),              'OM1_fx';    
        OM2_flux(MyLake_concentrations, MyLake_params),                                'OM2_fx';
        dissolved_bc(MyLake_concentrations('Pz'), 94971),                              'PO4_c';
        0.154,            'NO3_c'; %0 % Exact concentration for solute species
        14.7,             'FeOH3_fx'; % 14.7; %from Canavan et al AML
        0.638,            'SO4_c'; % 0.638; % Exact concentration for solute species
        0,                'Fe2_c'; %  0; % Exact concentration for solute species
        0,                'FeOOH_fx'; % 0; %from Canavan et al AML
        0,                'FeS_fx'; % 0; % Flux for solid species
        0,                'S0_c'; % 0 ; % Exact concentration for solute species
        0,                'S8_fx'; % 0; %from Canavan et al AML
        0,                'FeS2_fx'; % 0; % Flux for solid species
        0,                'AlOH3_fx'; % 0 % Flux for solid species
        0,                'PO4adsa_fx'; % 0; % Flux for solid species
        0,                'PO4adsb_fx'; % 0; % Flux for solid species
        0.04,             'Ca2_c'; % 0.04; % Exact concentration for solute species
        0,                'Ca3PO42_fx'; % 0; % Flux for solid species
        0,                'OMS_fx'; % 0; % Flux for solid species
        
        % pH module: for now assuming constant values satisfying pH=6.47 not taking into account reactions in WC of MyLake
        0.135974116,      'H_c'; 
        0.07354365,       'OH_c';
        9.960074871,      'CO2_c';
        2.19E-05,         'CO3_c';
        0.62387047,      'HCO3_c';
        3.68E-09,         'NH3_c';
        9.19E-07,         'NH4_c';
        1.01E-10,         'HS_c';
        1.06E-10,         'H2S_c';
        0.0169321276,     'H2CO3_c';
    };
    sediments_bc = containers.Map({sediments_bc{:,2}},{sediments_bc{:,1}});

  
end

function C_bc = dissolved_bc(C, M_C)
% return the value of boundary concentration for sediment
% C - concentration of the particular specie in the MyLake  [mg m-3]
% M_C - molar mass of this specie (need to convert [mg m-3]- > [umol/cm3]
    C_bc = C(end) / M_C;
end

function OM2_fx = OM2_flux(MyLake_concentrations, MyLake_params)
% this function convert H_netsed_catch to OM2 fluxes
% H_netsed_catch - flux of carbon from catchments (profile) [m day-1, dry]
% SS_C - Carbon fraction in H_netsed_catch [~]
% density_org_H_nc - Density of organic fraction in H_netsed_catch [g cm-3]
% OM2_fx - Sediments flux of OM2 [umol cm-2 yr-1]
    H_netsed_catch = MyLake_concentrations('H_netsed_catch');
    SS_C = MyLake_params('SS_C');
    density_org_H_nc = MyLake_params('density_org_H_nc');
    M_C = 12*1e-6; % Molar weight of carbon (g umol-1)
    OM2_fx = H_netsed_catch(end)*100*365*SS_C*density_org_H_nc/M_C; % 
end

function OM1_fx = OM1_flux(MyLake_concentrations, MyLake_params, sediments_params)
% function convert flux of Chl to OM1 flux in sediments
% Chl - chlorophyll (group 1) [mg m-3]
% w_chl - settling velocity for Chl a (m day-1)
% Cx1 - 
% fi -
% M_C - Molar mass of carbon [mg mol-1]
% Mass_Ratio_C_Chl - Fixed empirical ratio C:Chl (mass/mass)

    Chl = MyLake_concentrations('Chlz');
    w_chl = MyLake_params('w_chl') * 100 * 365; %settling velocity for Chl [m d-1] -> [cm year-1]   
    Cx1= sediments_params('Cx1');
    fi = sediments_params('fi');
    M_C = 12011; % Molar mass of carbon [mg mol-1]
    Mass_Ratio_C_Chl = MyLake_params('Mass_Ratio_C_Chl');

    OM1_fx =  (1 - fi(1)) * w_chl * Chl(end) * Mass_Ratio_C_Chl / ( M_C * Cx1);
end
