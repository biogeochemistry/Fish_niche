function [ sediment_results ] = single_run( )
% clear all; close all; clc;
%RUN Summary of this function goes here
%   Detailed explanation goes here

global sed_par_file


% Change here:
% ==============================================
T = 200; % how many days to run
% WC params:
Temperature = 8; % Temperature at SWI
z_max   = 12; % depth at SWI
pH_SWI = 5.2; % pH

params_sediment ...
    = {1,  'k_OM1';       
    0.01,  'k_OM2';         
    0.0123,'Km_O2';         
    0.01,  'Km_NO3';    
    3.92,  'Km_Fe(OH)3';    
    2415,  'Km_FeOOH';  
    0.0293,'Km_SO4';    
    0.001, 'Km_oxao';  
    0.1,   'Km_amao';    
    0.3292,'Kin_O2';  
    0.1,   'Kin_NO3';    
    0.1,   'Kin_FeOH3';  
    0.1,   'Kin_FeOOH';  
    2000,  'k_NH4ox';   
    8.7e4, 'k_Feox';    
    0.1,   'k_Sdis';    
    2500,  'k_Spre';   
    3.17,  'k_FeS2pre';  
    0.1,   'k_alum';    
    1.35,  'k_pdesorb_a';   
    1.35,  'k_pdesorb_b';   
    6500,  'k_rhom';    
    0.1,   'k_tS_Fe';   
    2510,  'Ks_FeS';    
    0.001, 'k_Fe_dis';  
    21.3,  'k_Fe_pre';  
    0.37,  'k_apa';         
    3e-6,  'kapa';  
    0.3134,'k_oms';     
    1000,  'k_tsox';    
    0.001, 'k_FeSpre';  
    30,    'accel';     
    1e-6,   'f_pfe';
    1.35,   'k_pdesorb_c';

    % Added porosity modeling parameters:
    0.98,   'fi_in'; 
    0.85,   'fi_f';
    0.5,    'X_b';
    1,      'tortuosity';

    10,      'w';
    64,     'n';
    15,     'depth';
    0.26,   'F';
    0,   'alfa0';
   
    % OM composition
    112,    'Cx1';
    10,     'Ny1';
    1,      'Pz1';
    200,    'Cx2';
    20,     'Ny2';
    1,      'Pz2';

    0.001,  'ts'};
% ===================================================


calibration_k_values = [(1:50)',cell2mat(params_sediment(:,1)) ]; % writing sediments   

sed_par_file = tempname;  
dlmwrite(sed_par_file, calibration_k_values,'delimiter','\t');



[sediment_concentrations, sediment_params, sediment_matrix_templates, species_sediments]  = sediments_init( pH_SWI, z_max, Temperature );

for i=1:T

    % Change here:
    % ==========================================================
    % You may specify the top Boundary conditions here. they can be varied as f(T):
    sediments_bc = {...
        0.15,      'Ox_c';
        60,        'OM1_fx';    
        30,        'OM2_fx';
        0.5,       'PO4_c';
        10,        'NO3_c';
        14.7,      'FeOH3_fx';
        10,        'SO4_c';
        10,        'Fe2_c';
        0,         'FeOOH_fx';
        0,         'FeS_fx';
        0,         'S0_c';
        0,         'S8_fx';
        0,         'FeS2_fx';
        0,         'AlOH3_fx';
        0,         'PO4adsa_fx';
        0,         'PO4adsb_fx';
        10,        'Ca2_c';
        0,         'Ca3PO42_fx';
        0,         'OMS_fx';

        % These values cannot be 0:
        0.1,      'H_c'; 
        0.1,      'OH_c';
        100,      'CO2_c';
        100,      'CO3_c';
        100,      'HCO3_c';
        0.1,      'NH3_c';
        0.1,      'NH4_c';
        0.1,      'HS_c';
        1E-10,    'H2S_c';
        100,      'H2CO3_c';
    };
    % ========================================================



    sediments_bc = containers.Map({sediments_bc{:,2}},{sediments_bc{:,1}});


    % Running sediment module
    [sediment_bioirrigation_fluxes, sediment_diffusion_fluxes, sediment_concentrations, z_matsedlab, R_values_matsedlabz] = sediments(...
    sediment_concentrations, sediment_params, sediment_matrix_templates, species_sediments, sediments_bc);


    % Output:
    O2_matsedlabzt(:,i) = sediment_concentrations('Oxygen');
    OM_matsedlabzt(:,i) = sediment_concentrations('OM1');
    OMb_matsedlabzt(:,i) = sediment_concentrations('OM2');
    NO3_matsedlabzt(:,i) = sediment_concentrations('NO3');
    FeOH3_matsedlabzt(:,i) = sediment_concentrations('FeOH3');
    SO4_matsedlabzt(:,i) = sediment_concentrations('SO4');
    NH4_matsedlabzt(:,i) = sediment_concentrations('NH4');
    Fe2_matsedlabzt(:,i) = sediment_concentrations('Fe2');
    FeOOH_matsedlabzt(:,i) = sediment_concentrations('FeOOH');
    H2S_matsedlabzt(:,i) = sediment_concentrations('H2S');
    HS_matsedlabzt(:,i)  = sediment_concentrations('HS');
    FeS_matsedlabzt(:,i) = sediment_concentrations('FeS');
    S0_matsedlabzt(:,i)  = sediment_concentrations('S0');
    PO4_matsedlabzt(:,i) = sediment_concentrations('PO4');
    S8_matsedlabzt(:,i) = sediment_concentrations('S8');
    FeS2_matsedlabzt(:,i) = sediment_concentrations('FeS2');
    AlOH3_matsedlabzt(:,i) = sediment_concentrations('AlOH3');
    PO4adsa_matsedlabzt(:,i) = sediment_concentrations('PO4adsa');
    PO4adsb_matsedlabzt(:,i) = sediment_concentrations('PO4adsb');
    Ca2_matsedlabzt(:,i) = sediment_concentrations('Ca2');
    Ca3PO42_matsedlabzt(:,i) = sediment_concentrations('Ca3PO42');
    OMS_matsedlabzt(:,i) = sediment_concentrations('OMS');
    H_matsedlabzt(:,i) = sediment_concentrations('H');
    OH_matsedlabzt(:,i) = sediment_concentrations('OH');
    CO2_matsedlabzt(:,i) = sediment_concentrations('CO2');
    CO3_matsedlabzt(:,i) = sediment_concentrations('CO3');
    HCO3_matsedlabzt(:,i) = sediment_concentrations('HCO3');
    NH3_matsedlabzt(:,i) = sediment_concentrations('NH3');
    H2CO3_matsedlabzt(:,i) = sediment_concentrations('H2CO3');
    pH_matsedlabzt(:,i) = -log10(H_matsedlabzt(:,i)*10^-6);
    O2_flux_matsedlabzt(i) = sediment_diffusion_fluxes{1};
    OM_flux_matsedlabzt(i) = sediment_diffusion_fluxes{2};
    OM2_flux_matsedlabzt(i) = sediment_diffusion_fluxes{3};
    PO4_flux_matsedlabzt(i) = sediment_diffusion_fluxes{4};
    R1_matsedlabzt(:,i) = R_values_matsedlabz{1};
    R1_int_matsedlabzt(:,i) = R_values_matsedlabz{2};
    R2_matsedlabzt(:,i) = R_values_matsedlabz{3};
    R2_int_matsedlabzt(:,i) = R_values_matsedlabz{4};
    R3_matsedlabzt(:,i) = R_values_matsedlabz{5};
    R3_int_matsedlabzt(:,i) = R_values_matsedlabz{6};
    R4_matsedlabzt(:,i) = R_values_matsedlabz{7};
    R4_int_matsedlabzt(:,i) = R_values_matsedlabz{8};
    R5_matsedlabzt(:,i) = R_values_matsedlabz{9};
    R5_int_matsedlabzt(:,i) = R_values_matsedlabz{10};
    O2_Bioirrigation_matsedlabz(:,i) = sediment_bioirrigation_fluxes{1};
    PO4_Bioirrigation_matsedlabz(:,i) = sediment_bioirrigation_fluxes{2};
    NO3_flux_matsedlabzt(i) = sediment_diffusion_fluxes{5};
    FeOH3_flux_matsedlabzt(i) = sediment_diffusion_fluxes{6};
    R6_matsedlabzt(:,i) = R_values_matsedlabz{11};
    R6_int_matsedlabzt(:,i) = R_values_matsedlabz{12};

end

R_values_matsedlabzt = { 

      R1_matsedlabzt,         'R1';
      R1_int_matsedlabzt,     'R1 integrated';
      R2_matsedlabzt,         'R2';
      R2_int_matsedlabzt,     'R2 integrated';
      R3_matsedlabzt,         'R3';
      R3_int_matsedlabzt,     'R3 integrated';
      R4_matsedlabzt,         'R4';
      R4_int_matsedlabzt,     'R4 integrated';
      R5_matsedlabzt,         'R5';
      R5_int_matsedlabzt,     'R5 integrated';
      R6_matsedlabzt,         'R6'; 
      R6_int_matsedlabzt,     'R6 integrated'; 
};

Bioirrigation_matsedlabzt = { O2_Bioirrigation_matsedlabz,  'Oxygen';
                              PO4_Bioirrigation_matsedlabz, 'PO4'};


sediment_results = {O2_matsedlabzt,     'Oxygen (aq)';
                   FeOH3_matsedlabzt,   'Iron hydroxide pool 1 Fe(OH)3 (s)';
                   FeOOH_matsedlabzt,   'Iron Hydroxide pool 2 FeOOH (s)';
                   SO4_matsedlabzt,     'Sulfate SO4(2-) (aq)';
                   Fe2_matsedlabzt,     'Iron Fe(2+) (aq)'; 
                   H2S_matsedlabzt,     'Sulfide H2S (aq)';
                   HS_matsedlabzt,      'Sulfide HS(-) (aq)'; 
                   FeS_matsedlabzt,     'Iron Sulfide FeS (s)';
                   OM_matsedlabzt,      'Organic Matter pool 1 OMa (s)';
                   OMb_matsedlabzt,     'Organic Matter pool 2 OMb (s)';
                   OMS_matsedlabzt,     'Sulfured Organic Matter (s)';
                   AlOH3_matsedlabzt,   'Aluminum oxide Al(OH)3 (s)'; 
                   S0_matsedlabzt,      'Elemental sulfur S(0) (aq)'; 
                   S8_matsedlabzt,      'Rhombic sulfur S8 (s)'; 
                   FeS2_matsedlabzt,    'Pyrite FeS2 (s)'; 
                   PO4_matsedlabzt,     'Phosphate PO4(3-) (aq)';
                   PO4adsa_matsedlabzt, 'Solid phosphorus pool a PO4adsa (s)';
                   PO4adsb_matsedlabzt, 'Solid phosphorus pool b PO4adsb (s)';
                   NO3_matsedlabzt,     'Nitrate NO3(-) (aq)';
                   NH4_matsedlabzt,     'Ammonium NH4(+) (aq)';
                   Ca2_matsedlabzt,     'Calcium Ca(2+) (aq)';
                   Ca3PO42_matsedlabzt, 'Apatite Ca3PO42 (s)';
                   H_matsedlabzt,       'H(+)(aq)';
                   OH_matsedlabzt,      'OH(-)(aq)';
                   CO2_matsedlabzt,     'CO2(aq)';
                   CO3_matsedlabzt,     'CO3(2-)(aq)';
                   HCO3_matsedlabzt,    'HCO3(-)(aq)';
                   NH3_matsedlabzt,     'NH3(aq)';
                   H2CO3_matsedlabzt,   'H2CO3(aq)';
                   pH_matsedlabzt,      'pH in sediment';
                   OM_flux_matsedlabzt, 'OM flux to sediments';
                   OM2_flux_matsedlabzt,'OM2 flux to sediments';
                   O2_flux_matsedlabzt, 'Oxygen flux WC to Sediments';
                   PO4_flux_matsedlabzt,'PO4 flux WC to Sediments';
                   NO3_flux_matsedlabzt,'NO3 flux to sediments';
                   FeOH3_flux_matsedlabzt,'FeOH3 flux to sediments';
                   z_matsedlab,         'z';
                   R_values_matsedlabzt,'R values of TEA oxidations sediments';
                   Bioirrigation_matsedlabzt, 'Fluxes of bioirrigation';
                   params_sediment,     'Sediments params';
};


end

