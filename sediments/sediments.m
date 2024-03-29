function [ sediment_bioirrigation_fluxes, sediment_diffusion_fluxes, sediment_concentrations, xz, R_values] = sediments(sediment_concentrations, sediment_params, sediment_matrix_templates, species_sediment, sediments_bc)
  % SEDIMENTS This function models the chemical process in the sediments based on Aguilera, 2005

  Ox_prev = sediment_concentrations('Oxygen');
  OM_prev = sediment_concentrations('OM1');
  OMb_prev = sediment_concentrations('OM2');
  NO3_prev = sediment_concentrations('NO3');
  FeOH3_prev = sediment_concentrations('FeOH3');
  SO4_prev = sediment_concentrations('SO4');
  NH4_prev = sediment_concentrations('NH4');
  Fe2_prev = sediment_concentrations('Fe2');
  FeOOH_prev = sediment_concentrations('FeOOH');
  H2S_prev = sediment_concentrations('H2S');
  HS_prev  = sediment_concentrations('HS');
  FeS_prev = sediment_concentrations('FeS');
  S0_prev  = sediment_concentrations('S0');
  PO4_prev = sediment_concentrations('PO4');
  S8_prev = sediment_concentrations('S8');
  FeS2_prev = sediment_concentrations('FeS2');
  AlOH3_prev = sediment_concentrations('AlOH3');
  PO4adsa_prev = sediment_concentrations('PO4adsa');
  PO4adsb_prev = sediment_concentrations('PO4adsb');
  Ca2_prev = sediment_concentrations('Ca2');
  Ca3PO42_prev = sediment_concentrations('Ca3PO42');
  OMS_prev = sediment_concentrations('OMS');
  OH_prev = sediment_concentrations('OH');
  CO2_prev = sediment_concentrations('CO2');
  CO3_prev = sediment_concentrations('CO3');
  HCO3_prev = sediment_concentrations('HCO3');
  NH3_prev = sediment_concentrations('NH3');
  H_prev = sediment_concentrations('H');
  H2CO3_prev = sediment_concentrations('H2CO3');


  % model domain:
  n  = sediment_params('n'); %points in spatial grid
  depth = sediment_params('depth'); %sediments depth
  years = sediment_params('years'); %1 day
  ts    = sediment_params('ts'); % time step
  % Scheme properties:
  alpha = sediment_params('alpha'); % Diffusion shift
  betta = sediment_params('betta'); % Advection shift 
  gama  = sediment_params('gama'); % Reaction shift 
  v = sediment_params('w'); % is time-dependent burial rate w = 0.1
  Db    = sediment_params('Db'); % is effective diffusion due to bioturbation, Canavan et al D_bio between 0-5, 5 in the top layers
  alfax = sediment_params('alfax');
  D_O2  = sediment_params('D_O2');
  D_NO3 = sediment_params('D_NO3');
  D_SO4 = sediment_params('D_SO4');
  D_NH4 = sediment_params('D_NH4');
  D_Fe2 = sediment_params('D_Fe2');
  D_H2S = sediment_params('D_H2S');
  D_S0  = sediment_params('D_S0');
  D_PO4 = sediment_params('D_PO4');
  D_Ca2 = sediment_params('D_Ca2');
  D_HS  = sediment_params('D_HS');
  Cx1   = sediment_params('Cx1');
  Ny1   = sediment_params('Ny1');
  Pz1   = sediment_params('Pz1');
  Cx2   = sediment_params('Cx2');
  Ny2   = sediment_params('Ny2');
  Pz2   = sediment_params('Pz2');
  fi    = sediment_params('fi');
  F     = sediment_params('F');  %conversion factor = rhob * (1-fi) / fi ; where fi = porosity and rhob = solid phase density

  K_OM =  sediment_params('k_OM');
  K_OMb = sediment_params('k_OMb');
  Km_O2 = sediment_params('Km_O2');
  Km_NO3 = sediment_params('Km_NO3');
  Km_FeOH3 = sediment_params('Km_FeOH3');
  Km_FeOOH = sediment_params('Km_FeOOH');
  Km_SO4 = sediment_params('Km_SO4');
  Km_oxao = sediment_params('Km_oxao');
  Km_amao = sediment_params('Km_amao');
  Kin_O2 = sediment_params('Kin_O2');
  Kin_NO3  = sediment_params('Kin_NO3');
  Kin_FeOH3 = sediment_params('Kin_FeOH3');
  Kin_FeOOH = sediment_params('Kin_FeOOH');
  k_amox = sediment_params('k_amox');
  k_Feox = sediment_params('k_Feox');
  k_Sdis = sediment_params('k_Sdis');
  k_Spre = sediment_params('k_Spre');
  k_FeS2pre = sediment_params('k_FeS2pre');
  k_pdesorb_c = sediment_params('k_pdesorb_c');
  k_pdesorb_a = sediment_params('k_pdesorb_a');
  k_pdesorb_b = sediment_params('k_pdesorb_b');
  k_alum = sediment_params('k_alum');
  k_rhom   = sediment_params('k_rhom');
  k_tS_Fe = sediment_params('k_tS_Fe');
  Ks_FeS = sediment_params('Ks_FeS');
  k_Fe_dis = sediment_params('k_Fe_dis');
  k_Fe_pre = sediment_params('k_Fe_pre');
  k_apa  = sediment_params('k_apa');
  kapa = sediment_params('kapa');
  k_oms = sediment_params('k_oms');
  k_tsox = sediment_params('k_tsox');
  k_FeSpre = sediment_params('k_FeSpre');
  f_pfe = sediment_params('f_pfe');
  accel = sediment_params('accel');
  x  = sediment_params('x');
  
  dx = x(2)-x(1);
  xz = x'/100;
  % time domain:
  t     = 0:ts:years; % years
  m     = size(t,2); %steps in time
  dt    = t(2)-t(1);

    % Chemical constants
  % =======================================================================================================

  P_index1 = Pz1/Cx1;
  P_index2 = Pz2/Cx2;
  N_index1 = Ny1/Cx1;
  N_index2 = Ny2/Cx2;

  

  % Allocation of the memory and formation of template matrix:
  % =======================================================================================================

  % Solid species: row #1 of the cell "sediment_matrix_templates" is the solid template matrix
  [LU_om0,  RK_om0,  LD_om0,  LA_om0,  RD_om0,  RA_om0] = sediment_matrix_templates{1,1:6}; 
  [LU_omb0, RK_omb0, LD_omb0, LA_omb0, RD_omb0, RA_omb0] = sediment_matrix_templates{1,1:6};
  [LU_FeOOH0, RK_FeOOH0, LD_FeOOH0, LA_FeOOH0, RD_FeOOH0, RA_FeOOH0] = sediment_matrix_templates{1,1:6};
  [LU_FeS0, RK_FeS0, LD_FeS0, LA_FeS0, RD_FeS0, RA_FeS0] = sediment_matrix_templates{1,1:6};
  [LU_S80, RK_S80, LD_S80, LA_S80, RD_S80, RA_S80] = sediment_matrix_templates{1,1:6};
  [LU_FeS20, RK_FeS20, LD_FeS20, LA_FeS20, RD_FeS20, RA_FeS20] = sediment_matrix_templates{1,1:6};
  [LU_AlOH30, RK_alum0, LD_AlOH30, LA_AlOH30, RD_AlOH30, RA_AlOH30] = sediment_matrix_templates{1,1:6};
  [LU_Ca3PO420, RK_Ca3PO420, LD_Ca3PO420, LA_Ca3PO420, RD_Ca3PO420, RA_Ca3PO420] = sediment_matrix_templates{1,1:6};
  [LU_PO4adsa0, RK_PO4adsa0, LD_PO4adsa0, LA_PO4adsa0, RD_PO4adsa0, RA_PO4adsa0] = sediment_matrix_templates{1,1:6};
  [LU_PO4adsb0, RK_PO4adsb0, LD_PO4adsb0, LA_PO4adsb0, RD_PO4adsb0, RA_PO4adsb0] = sediment_matrix_templates{1,1:6};
  [LU_OMS0, RK_OMS0, LD_OMS0, LA_OMS0, RD_OMS0, RA_OMS0] = sediment_matrix_templates{1,1:6};
  [LU_FeOH30, RK_FeOH30, LD_FeOH30, LA_FeOH30, RD_FeOH30, RA_FeOH30] = sediment_matrix_templates{1,1:6};

  % Solute species:
  [LU_ox0,  RK_ox0,  LD_ox0,  LA_ox0,  RD_ox0,  RA_ox0] = sediment_matrix_templates{2,1:6};
  [LU_NO30, RK_NO30, LD_NO30, LA_NO30, RD_NO30, RA_NO30] = sediment_matrix_templates{3,1:6};
  [LU_SO40, RK_SO40, LD_SO40, LA_SO40, RD_SO40, RA_SO40] = sediment_matrix_templates{4,1:6};
  [LU_NH40, RK_NH40, LD_NH40, LA_NH40, RD_NH40, RA_NH40] = sediment_matrix_templates{5,1:6};
  [LU_Fe20, RK_Fe20, LD_Fe20, LA_Fe20, RD_Fe20, RA_Fe20] = sediment_matrix_templates{6,1:6};
  [LU_H2S0, RK_H2S0, LD_H2S0, LA_H2S0, RD_H2S0, RA_H2S0] = sediment_matrix_templates{7,1:6};
  [LU_S00, RK_S00, LD_S00, LA_S00, RD_S00, RA_S00] = sediment_matrix_templates{8,1:6};
  [LU_PO40, RK_PO40, LD_PO40, LA_PO40, RD_PO40, RA_PO40] = sediment_matrix_templates{9,1:6};
  [LU_Ca20, RK_Ca20, LD_Ca20, LA_Ca20, RD_Ca20, RA_Ca20] = sediment_matrix_templates{10,1:6};
  [LU_HS0, RK_HS0, LD_HS0, LA_HS0, RD_HS0, RA_HS0] = sediment_matrix_templates{11,1:6};
  [LU_H0, RK_H0, LD_H0, LA_H0, RD_H0, RA_H0] = sediment_matrix_templates{12,1:6};
  [LU_OH0, RK_OH0, LD_OH0, LA_OH0, RD_OH0, RA_OH0] = sediment_matrix_templates{13,1:6};
  [LU_CO20, RK_CO20, LD_CO20, LA_CO20, RD_CO20, RA_CO20] = sediment_matrix_templates{14,1:6};
  [LU_CO30, RK_CO30, LD_CO30, LA_CO30, RD_CO30, RA_CO30] = sediment_matrix_templates{15,1:6};
  [LU_HCO30, RK_HCO30, LD_HCO30, LA_HCO30, RD_HCO30, RA_HCO30] = sediment_matrix_templates{16,1:6};
  [LU_NH30, RK_NH30, LD_NH30, LA_NH30, RD_NH30, RA_NH30] = sediment_matrix_templates{17,1:6};
  [LU_H2CO30, RK_H2CO30, LD_H2CO30, LA_H2CO30, RD_H2CO30, RA_H2CO30] = sediment_matrix_templates{18,1:6};



  % Allocation of the memory for BC matrices (it is not equal to boundary conditions)
  % =======================================================================================================
  BC_Ox_matrix    = zeros(n-1,1); 
  BC_OM_matrix    = zeros(n,1); 
  BC_OMb_matrix   = zeros(n,1);
  BC_NO3_matrix   = zeros(n-1,1);
  BC_FeOH3_matrix = zeros(n,1);
  BC_SO4_matrix   = zeros(n-1,1);
  BC_NH4_matrix   = zeros(n-1,1);
  BC_Fe2_matrix   = zeros(n-1,1);
  BC_FeOOH_matrix = zeros(n,1);
  BC_H2S_matrix   = zeros(n-1,1);
  BC_FeS_matrix   = zeros(n,1);
  BC_S0_matrix    = zeros(n-1,1);
  BC_PO4_matrix   = zeros(n-1,1);
  BC_S8_matrix    = zeros(n,1);
  BC_FeS2_matrix  = zeros(n,1);
  BC_AlOH3_matrix = zeros(n,1);
  BC_PO4adsa_matrix = zeros(n,1);
  BC_PO4adsb_matrix = zeros(n,1);
  BC_Ca2_matrix   = zeros(n-1,1);
  BC_Ca3PO42_matrix = zeros(n,1);
  BC_OMS_matrix   = zeros(n,1);
  BC_HS_matrix    = zeros(n-1,1);
  BC_H_matrix     = zeros(n-1,1);
  BC_OH_matrix    = zeros(n-1,1);
  BC_CO2_matrix   = zeros(n-1,1);
  BC_CO3_matrix   = zeros(n-1,1);
  BC_HCO3_matrix  = zeros(n-1,1);
  BC_NH3_matrix   = zeros(n-1,1);
  BC_H2CO3_matrix = zeros(n-1,1);

  % Allocation of the memory for of R_eq
  % =======================================================================================================
  R_eq_Ox_non_lin = zeros(n,m);
  R_eq_OM_non_lin  = zeros(n,m);
  R_eq_OMb_non_lin = zeros(n,m);
  R_eq_NO3_non_lin = zeros(n,m);
  R_eq_FeOH3_non_lin = zeros(n,m);
  R_eq_SO4_non_lin = zeros(n,m);
  R_eq_NH4_non_lin = zeros(n,m);
  R_eq_Fe2_non_lin = zeros(n,m);
  R_eq_FeOOH_non_lin  = zeros(n,m);
  R_eq_H2S_non_lin = zeros(n,m);
  R_eq_FeS_non_lin = zeros(n,m);
  R_eq_S0_non_lin  = zeros(n,m);
  R_eq_PO4_non_lin = zeros(n,m);
  R_eq_S8_non_lin  = zeros(n,m);
  R_eq_FeS2_non_lin= zeros(n,m);
  R_eq_AlOH3_non_lin= zeros(n,m);
  R_eq_PO4adsa_non_lin= zeros(n,m);
  R_eq_PO4adsb_non_lin= zeros(n,m);
  R_eq_Ca2_non_lin = zeros(n,m);
  R_eq_Ca3PO42_non_lin= zeros(n,m);
  R_eq_OMS_non_lin= zeros(n,m);
  R_eq_HS_non_lin = zeros(n,m);
  R_eq_H_non_lin = zeros(n,m);
  R_eq_OH_non_lin = zeros(n,m);
  R_eq_CO2_non_lin = zeros(n,m);
  R_eq_CO3_non_lin = zeros(n,m);
  R_eq_HCO3_non_lin = zeros(n,m);
  R_eq_NH3_non_lin = zeros(n,m);
  R_eq_H2CO3_non_lin = zeros(n,m);
  
  R_eq_OM = zeros(n,m);
  R_eq_OMb = zeros(n,m);
  R_eq_oxygen = zeros(n,m);
  R_eq_NO3 = zeros(n,m);
  R_eq_SO4 = zeros(n,m);
  R_eq_NH4 = zeros(n,m);
  R_eq_Fe2 = zeros(n,m);
  R_eq_FeOH3 = zeros(n,m);
  R_eq_FeOOH = zeros(n,m);
  R_eq_H2S = zeros(n,m);
  R_eq_FeS = zeros(n,m);
  R_eq_S0 = zeros(n,m);
  R_eq_PO4 = zeros(n,m);
  R_eq_S8 = zeros(n,m);
  R_eq_FeS2 = zeros(n,m);
  R_eq_AlOH3 = zeros(n,m);
  R_eq_PO4adsa = zeros(n,m);
  R_eq_PO4adsb = zeros(n,m);
  R_eq_Ca2 = zeros(n,m);
  R_eq_Ca3PO42 = zeros(n,m);
  R_eq_OMS = zeros(n,m);
  R_eq_HS = zeros(n,m);
  R_eq_H = zeros(n,m);
  R_eq_OH = zeros(n,m);
  R_eq_CO2 = zeros(n,m);
  R_eq_CO3 = zeros(n,m);
  R_eq_HCO3 = zeros(n,m);
  R_eq_NH3 = zeros(n,m);
  R_eq_H2CO3 = zeros(n,m);
  
  R1_graphs = zeros(n,m-1);
  R2_graphs = zeros(n,m-1);
  R3_graphs = zeros(n,m-1);
  R4_graphs = zeros(n,m-1);
  R5_graphs = zeros(n,m-1);
  R6_graphs = zeros(n,m-1);
  
  % For debugging
  % difference = zeros(n,m);
  Sat_FeS = zeros(n,m);

  % Constructing BC (boundary conditions):
  % =======================================================================================================

  % for Oxygen:
  BC_O_top = ones(1,m) * sediments_bc('Ox_c'); % Exact concentration for solute species
  BC_O = zeros(n-1,m);
  BC_O(1,:) = BC_O_top;

  % for Organic matter A:
  % Flux of OM1 from Lake
  F_OM_top = ones(1,m) * sediments_bc('OM1_fx'); % Flux on the top for solid species
  
  % for Organic matter B:
  % Flux of OM2 from Lake
  F_OMb_top = ones(1,m) * sediments_bc('OM2_fx'); % Flux on the top for solid species

  % For NO3
  BC_NO3_top = ones(1,m) * sediments_bc('NO3_c'); %0 % Exact concentration for solute species
  BC_NO3 = zeros(n-1,m);
  BC_NO3(1,:) = BC_NO3_top;

  % for Fe(OH)3
  F_FeOH3_top = ones(1,m) * sediments_bc('FeOH3_fx'); % 14.7; %from Canavan et al AML

  % For SO4
  BC_SO4_top = ones(1,m) * sediments_bc('SO4_c'); % 0.638; % Exact concentration for solute species
  BC_SO4 = zeros(n-1,m);
  BC_SO4(1,:) = BC_SO4_top;


  % For Fe2
  BC_Fe2_top = ones(1,m) * sediments_bc('Fe2_c'); %  0; % Exact concentration for solute species
  BC_Fe2 = zeros(n-1,m);
  BC_Fe2(1,:) = BC_Fe2_top;

  % for FeOOH
  F_FeOOH_top = ones(1,m) * sediments_bc('FeOOH_fx'); % 0; %from Canavan et al AML


  % for FeS
  F_FeS_top = ones(1,m) * sediments_bc('FeS_fx'); % 0; % Flux for solid species

  % For S(0)
  BC_S0_top = ones(1,m) * sediments_bc('S0_c'); % 0 ; % Exact concentration for solute species
  BC_S0 = zeros(n-1,m);
  BC_S0(1,:) = BC_S0_top;

  % For PO4
  BC_PO4_top = ones(1,m) * sediments_bc('PO4_c'); % Exact concentration for solute species
  BC_PO4 = zeros(n-1,m);
  BC_PO4(1,:) = BC_PO4_top;

  % for S8
  F_S8_top = ones(1,m) * sediments_bc('S8_fx'); % 0; %from Canavan et al AML

  % for FeS2
  F_FeS2_top = ones(1,m) * sediments_bc('FeS2_fx'); % 0; % Flux for solid species

  % for AlOH3
  F_AlOH3_top = ones(1,m) * sediments_bc('AlOH3_fx'); % 0 % Flux for solid species

  % for PO4adsa
  F_PO4adsa_top = ones(1,m) * sediments_bc('PO4adsa_fx'); % 0; % Flux for solid species

  % for PO4adsb
  F_PO4adsb_top = ones(1,m) * sediments_bc('PO4adsb_fx'); % 0; % Flux for solid species

  % For Ca2
  BC_Ca2_top = ones(1,m) * sediments_bc('Ca2_c'); % 0.04; % Exact concentration for solute species
  BC_Ca2 = zeros(n-1,m);
  BC_Ca2(1,:) = BC_Ca2_top;

  % for Ca3PO42
  F_Ca3PO42_top = ones(1,m) * sediments_bc('Ca3PO42_fx'); % 0; % Flux for solid species

  % for OMS
  F_OMS_top = ones(1,m) .* sediments_bc('OMS_fx'); % 0; % Flux for solid species

  % For H
  BC_H_top = ones(1,m) * sediments_bc('H_c'); % pH = 6.47 Exact concentration for solute species
  BC_H = zeros(n-1,m);
  BC_H(1,:) = BC_H_top;

  % For OH
  BC_OH_top = ones(1,m) * sediments_bc('OH_c'); % Exact concentration for solute species
  BC_OH = zeros(n-1,m);
  BC_OH(1,:) = BC_OH_top;

  % For CO2
  BC_CO2_top = ones(1,m) * sediments_bc('CO2_c'); % Exact concentration for solute species
  BC_CO2 = zeros(n-1,m);
  BC_CO2(1,:) = BC_CO2_top;

  % For CO3
  BC_CO3_top = ones(1,m) * sediments_bc('CO3_c'); % Exact concentration for solute species
  BC_CO3 = zeros(n-1,m);
  BC_CO3(1,:) = BC_CO3_top;

  % For HCO3
  BC_HCO3_top = ones(1,m) * sediments_bc('HCO3_c'); % Exact concentration for solute species
  BC_HCO3 = zeros(n-1,m);
  BC_HCO3(1,:) = BC_HCO3_top;

  % For NH3
  BC_NH3_top = ones(1,m) * sediments_bc('NH3_c'); % Exact concentration for solute species
  BC_NH3 = zeros(n-1,m);
  BC_NH3(1,:) = BC_NH3_top;

  % For NH4
  BC_NH4_top = ones(1,m) * sediments_bc('NH4_c'); % Exact concentration for solute species
  BC_NH4 = zeros(n-1,m);
  BC_NH4(1,:) = BC_NH4_top;

  % For S(-I)
  BC_HS_top = ones(1,m) * sediments_bc('HS_c'); % Exact concentration for solute species
  BC_HS = zeros(n-1,m);
  BC_HS(1,:) = BC_HS_top;

  % For S(-II)
  BC_H2S_top = ones(1,m) * sediments_bc('H2S_c'); % Exact concentration for solute species
  BC_H2S = zeros(n-1,m);
  BC_H2S(1,:) = BC_H2S_top;

  % For H2CO3
  BC_H2CO3_top = ones(1,m) * sediments_bc('H2CO3_c'); % Exact concentration for solute species
  BC_H2CO3 = zeros(n-1,m);
  BC_H2CO3(1,:) = BC_H2CO3_top;

  % Allocation of the memory for concentration with initial condition: (umol/cm3) or (umol/g)
  % ===============================================================
  % NOTE: All columns consist of init vector of concentrations

  Ox = ones(n,m); 
  Ox = bsxfun(@times,Ox,Ox_prev); 
  Ox(1,:) = BC_O_top;

  OM = ones(n,m); 
  OM = bsxfun(@times,OM,OM_prev); 

  OMb = ones(n,m); 
  OMb = bsxfun(@times,OMb,OMb_prev); 

  NO3 = ones(n,m); 
  NO3 = bsxfun(@times,NO3,NO3_prev); 
  NO3(1,:) = BC_NO3_top;

  FeOH3 = ones(n,m); 
  FeOH3 = bsxfun(@times,FeOH3,FeOH3_prev); 

  SO4 = ones(n,m); 
  SO4 = bsxfun(@times,SO4,SO4_prev); 
  SO4(1,:) = BC_SO4_top;

  Fe2 = ones(n,m); 
  Fe2 = bsxfun(@times,Fe2,Fe2_prev); 
  Fe2(1,:) = BC_Fe2_top;

  FeOOH = ones(n,m); 
  FeOOH = bsxfun(@times,FeOOH,FeOOH_prev); 

  FeS = ones(n,m); 
  FeS = bsxfun(@times,FeS,FeS_prev); 

  S0 = ones(n,m); 
  S0 = bsxfun(@times,S0,S0_prev); 
  S0(1,:) = BC_S0_top;

  PO4 = ones(n,m); 
  PO4 = bsxfun(@times,PO4,PO4_prev); 
  PO4(1,:) = BC_PO4_top;

  S8 = ones(n,m); 
  S8 = bsxfun(@times,S8,S8_prev); 

  FeS2 = ones(n,m); 
  FeS2 = bsxfun(@times,FeS2,FeS2_prev); 

  AlOH3 = ones(n,m); 
  AlOH3 = bsxfun(@times,AlOH3,AlOH3_prev); 

  PO4adsa = ones(n,m); 
  PO4adsa = bsxfun(@times,PO4adsa,PO4adsa_prev); 

  PO4adsb = ones(n,m); 
  PO4adsb(:,1) = PO4adsb_prev;

  Ca2 = ones(n,m); 
  Ca2 = bsxfun(@times,Ca2,Ca2_prev); 
  Ca2(1,:) = BC_Ca2_top;

  Ca3PO42 = ones(n,m); 
  Ca3PO42 = bsxfun(@times,Ca3PO42,Ca3PO42_prev); 

  OMS = ones(n,m); 
  OMS = bsxfun(@times,OMS,OMS_prev); 

  H = ones(n,m); 
  H = bsxfun(@times,H,H_prev); 
  H(1,:) = BC_H_top;

  OH = ones(n,m); 
  OH = bsxfun(@times,OH,OH_prev); 
  OH(1,:) = BC_OH_top;

  CO2 = ones(n,m); 
  CO2 = bsxfun(@times,CO2,CO2_prev); 
  CO2(1,:) = BC_CO2_top;

  CO3 = ones(n,m); 
  CO3 = bsxfun(@times,CO3,CO3_prev); 
  CO3(1,:) = BC_CO3_top;

  HCO3 = ones(n,m); 
  HCO3 = bsxfun(@times,HCO3,HCO3_prev); 
  HCO3(1,:) = BC_HCO3_top;


  NH3 = ones(n,m); 
  NH3 = bsxfun(@times,NH3,NH3_prev); 
  NH3(1,:) = BC_NH3_top;

  NH4 = ones(n,m); 
  NH4 = bsxfun(@times,NH4,NH4_prev); 
  NH4(1,:) = BC_NH4_top;

  HS = ones(n,m); 
  HS = bsxfun(@times,HS,HS_prev); 
  HS(1,:) = BC_HS_top;

  H2S = ones(n,m); 
  H2S = bsxfun(@times,H2S,H2S_prev); 
  H2S(1,:) = BC_H2S_top;

  H2CO3 = ones(n,m); 
  H2CO3 = bsxfun(@times,H2CO3,H2CO3_prev); 
  H2CO3(1,:) = BC_H2CO3_top;


  % Stiffness:
  % =======================================================================================================
  % CFL = (D_O2+Db)*dt/dx^2;


  % Solving equations!!!
  % =========================================================================================================

  for i=2:m
    % NOTE: we can decrease the amount of computation by changing the order avoiding repeated computation of the same value
    % ============= Reduction coefficients ==================================================================
    % without concentration of species in the numerator, it will be multiplied by C inside of the matrices; 
    f_O2    =                                                                                                                                                               1 ./  (Km_O2 + Ox(:,i-1)) ; 
    f_NO3   =                                                                                                                           1 ./  (Km_NO3 + NO3(:,i-1)) .* Kin_O2 ./ (Kin_O2 + Ox(:,i-1)) ;
    f_FeOH3 =                                                                                  1 ./  (Km_FeOH3 + FeOH3(:,i-1)) .* Kin_NO3 ./ (Kin_NO3 + NO3(:,i-1)) .* Kin_O2 ./ (Kin_O2 + Ox(:,i-1)) ;
    f_FeOOH =                                       1 ./  (Km_FeOOH + FeOOH(:,i-1)) .* Kin_FeOH3 ./ (Kin_FeOH3 + FeOH3(:,i-1)) .* Kin_NO3 ./ (Kin_NO3 + NO3(:,i-1)) .* Kin_O2 ./ (Kin_O2 + Ox(:,i-1)) ;
    f_SO4 = 1 ./ (Km_SO4 + SO4 (:, i-1)) .* Kin_FeOOH ./ (Kin_FeOOH + FeOOH(:,i-1)) .* Kin_FeOH3 ./ (Kin_FeOH3 + FeOH3(:,i-1)) .* Kin_NO3 ./ (Kin_NO3 + NO3(:,i-1)) .* Kin_O2 ./ (Kin_O2 + Ox(:,i-1)) ;
    Sum_H2S = H2S(:,i-1) + HS(:,i-1);

    % ============= OMa =====================================================================================
    % R_OMa = -(R1a+R2a+R3a+R5a) - R10;  

    % All variables R_eq are without C(ox) in numerator due to it is will be added by matrix multiplication
    Rc1 = K_OM; % without OM concentration in the numerator, it will be multiplied by C(OM) inside of the OM matrix; 
    R1a = Rc1 * f_O2 .* Ox(:,i-1) *accel;            %OM oxidation
    R2a = Rc1 * f_NO3 .* NO3(:,i-1);          %OM oxidation by NO3 
    R3a = Rc1 * f_FeOH3 .* FeOH3(:,i-1) ;         %OM oxidation by Fe(OH)3
    R4a = Rc1 * f_FeOOH .* FeOOH(:,i-1);          %OM oxidation by FeOOH
    R5a = Rc1 * f_SO4 .* SO4(:,i-1) ;             %OM oxidation by SO4
    Ra = R1a+R2a+R3a+R4a+R5a;                     % degradation of OM1

    R10 = k_oms * Sum_H2S;
    R_eq_OM(:,i) = -Ra - R10;

    % In R_eq_non_lin should be all variables due to it is outside matrix multiplication
    R_eq_OM_non_lin(:,i) = 0;

    % ============= OMb =====================================================================================
    % R_OMb = -(R1b+R2b+R3b+R5b) - R10;   

    % All variables R_eq are without C(OMb) in numerator due to it is will be added by matrix multiplication
    Rc2 = K_OMb;  % without OM concentration in the numerator, it will be multiplied by C(OM) inside of the OM matrix; 
    R1b = Rc2 * f_O2 .* Ox(:,i-1) *accel;            %OM oxidation
    R2b = Rc2 * f_NO3 .* NO3(:,i-1);              %OM oxidation by NO3 
    R3b = Rc2 * f_FeOH3 .* FeOH3(:,i-1) ;         %OM oxidation by Fe(OH)3
    R4b = Rc2 * f_FeOOH .* FeOOH(:,i-1);          %OM oxidation by FeOOH
    R5b = Rc2 * f_SO4 .* SO4(:,i-1);              %OM oxidation by SO4
    Rb = R1b+R2b+R3b+R4b+R5b;                     % degradation of OM2
    
    R10 = k_oms * Sum_H2S;
    R_eq_OMb(:,i) = -Rb - R10;
    % In R_eq_non_lin should be all variables due to it is outside matrix multiplication
    R_eq_OMb_non_lin(:,i) = 0;

    
    % ============= Oxygen ==================================================================================
    % R_Ox = -R1*F - R6  - 0.25*R8 - 3*R12 - 2*R9

    % All variables R_eq are without C(Ox) in numerator due to it is will be added by matrix multiplication
    R8 = k_Feox .* Fe2(:,i-1);
    R9 = k_amox .* 1 ./ (Km_oxao + Ox(:,i-1)) .* (NH4(:,i-1) ./ (Km_amao + NH4(:,i-1)));
    R12 = k_rhom  .* FeS(:,i-1);
    Rc1 = K_OM * OM(:,i-1);                       % with OM concentration in the numerator 
    Rc2 = K_OMb * OMb(:,i-1);  
    R1a = Rc1 .* f_O2 * accel;                      %OM oxidation
    R1b = Rc2 .* f_O2 * accel;                      %OM oxidation
    R1 = R1a+R1b;                                %OM oxidation
    R_eq_oxygen(:,i) = -0.25 * R8  - 2 * R9  - R1 * F - 3 * R12;
    % In R_eq_non_lin should be all variables due to it is outside matrix multiplication
    Ox_dCbio(:,i) = bioirrigation(Ox(:,i-1), alfax, fi);
    R_eq_Ox_non_lin(:,i) = Ox_dCbio(:,i);
    
    % ============= NO3 =====================================================================================
    % R_NO3 = - 0.8*R2*F + R9;

    % All variables R_eq are without C(NO3) in numerator due to it is will be added by matrix multiplication
    R2b = Rc2 .* f_NO3;                     %OM oxidation by NO3 
    R2a = Rc1 .* f_NO3;                     %OM oxidation by NO3
    R2 = R2a+R2b;                                %OM oxidation by NO3
    R_eq_NO3(:,i) = - 0.8*R2*F;
    % In R_eq_non_lin should be all variables due to it is outside matrix multiplication
    R9 = k_amox * Ox(:,i-1) ./ (Km_oxao + Ox(:,i-1)) .* (NH4(:,i-1) ./ (Km_amao + NH4(:,i-1))); %ammonia oxidation with MM kinetics AML
    R_eq_NO3_non_lin(:,i) = R9 + bioirrigation(NO3(:,i-1), alfax, fi);

     
    % ============= SO4 ===================================================================================== 
    % R_SO4 = - R5*F + R6 

    % All variables R_eq are without C(SO4) in numerator due to it is will be added by matrix multiplication
    R5a = Rc1 .* f_SO4;                          %OM oxidation by SO4
    R5b = Rc2 .* f_SO4;                          %OM oxidation by SO4
    R5 = R5a+R5b;                                %OM oxidation by SO4
    R_eq_SO4(:,i) = R5 * F;
    % In R_eq_non_lin should be all variables due to it is outside matrix multiplication
    R6 = k_tsox * Ox(:,i-1) .* Sum_H2S;
    R_eq_SO4_non_lin(:,i) = R6 + bioirrigation(SO4(:,i-1), alfax, fi);
    
    % ============= Fe2+ ====================================================================================
    % R_Fe2 = 4*R3*F + 4*R4*F + 2*R7 - R8 + R14b - R14a 

    % All variables here are without C(Fe[2+]) in numerator due to it is will be added by matrix multiplication
    R8 = k_Feox .* Ox(:,i-1);
    R_eq_Fe2(:,i) =  - R8 ;

    % In R_eq_non_lin should be all variables due to it is outside matrix multiplication
    R3a = Rc1 .* f_FeOH3 .* FeOH3(:,i-1);          %OM oxidation by Fe(OH)3
    R3b = Rc2 .* f_FeOH3 .* FeOH3(:,i-1);          %OM oxidation by Fe(OH)3
    R3 = R3a+R3b; 
    R4a = Rc1 .* f_FeOOH .* FeOOH(:,i-1);          %OM oxidation by FeOOH
    R4b = Rc2 .* f_FeOOH .* FeOOH(:,i-1);           %OM oxidation by FeOOH
    R4 = R4a + R4b;
     %TODO: do we need to implement [HS(-)]?
    Sat_FeS(:,i-1) = Fe2(:,i-1) .* Sum_H2S ./ (H(:,i-1).^2 .* Ks_FeS);
    R14a = k_Fe_pre .* ( Sat_FeS(:,i-1) - 1);
    R14b  = k_Fe_dis .* FeS(:,i-1) .* ( 1 - Sat_FeS(:,i-1));
    R14a = (R14a >= 0) .* R14a; % can  be only non negative
    R14b  = (R14b >= 0) .* R14b; % can  be only non negative
    R7 = k_tS_Fe * FeOH3(:,i-1) .*  Sum_H2S;
    R_eq_Fe2_non_lin(:,i) = 4*R3*F + 4*R4*F + 2*R7 + R14b - R14a + bioirrigation(Fe2(:,i-1), alfax, fi); %NOTE: concentration of Iron in Sat_FeS can be improved by implementing and adding in R_eq_matrix


    % ============= FeOOH ====================================================================================
    % R_FeOOH = -4*R4  - R17a + R12
    R4a = Rc1 .* f_FeOOH;                   
    R4b = Rc2 .* f_FeOOH;                    
    R4 = R4a + R4b;

    R12 = k_rhom * Ox(:,i-1) .* FeS(:,i-1);
    R17a = k_pdesorb_b .* PO4(:,i-1);

    R_eq_FeOOH(:,i) = -4*R4 - R17a;
    R_eq_FeOOH_non_lin(:,i) = R12;

    % ============= Fe(OH)3 =================================================================================
    % R_Fe(OH)3 = - 4*R3 - 2*R7  + R8 -R16a

    % All variables R_eq are without C(Fe(OH)3) in numerator due to it is will be added by matrix multiplication
    R3a = Rc1 .* f_FeOH3;                        %OM oxidation by Fe(OH)3
    R3b = Rc2 .* f_FeOH3;                        %OM oxidation by Fe(OH)3
    R3 = R3a+R3b;                                %OM oxidation by Fe(OH)3
    R16a = k_pdesorb_a .* PO4(:,i-1);   
    R7 = k_tS_Fe .*  Sum_H2S;          
    R_eq_FeOH3(:,i) = -4 * R3 - R16a - 2*R7;
    % In R_eq_non_lin should be all variables due to it is outside matrix multiplication
    R8 = k_Feox .* Fe2(:,i-1) .* Ox(:,i-1);
    R_eq_FeOH3_non_lin(:,i) = R8;


    % ============= FeS ====================================================================================
    % R_FeS = -R14b + R14a - R11 -4*R12 - R13

    R12 = k_rhom * Ox(:,i-1);
    Sat_FeS(:,i-1) = Fe2(:,i-1) .* Sum_H2S ./ (H(:,i-1).^2 .* Ks_FeS);

    R14a = k_Fe_pre .* ( Sat_FeS(:,i-1) - 1);
    R14b  = k_Fe_dis .* ( 1 - Sat_FeS(:,i-1));
    R14a = (R14a >= 0) .* R14a; % can  be only non negative
    R14b  = (R14b >= 0) .* R14b; % can  be only non negative
    R11 = k_FeSpre .* S0(:,i-1); 
    R13 = k_FeS2pre .* Sum_H2S;
    R_eq_FeS(:,i) = - R14b - R11 - 4*R12 -R13;
    R_eq_FeS_non_lin(:,i) = R14a;


    % ============= PO4 ====================================================================================
    % R_PO4 =  (P_index1 * Ra + P_index2 * Rb)*F - R18a + R18b - R16a - R17a + R16b + R17b - 2*R19
    % TODO: add R18b
    % Matsedlab model
    R1a = Rc1 .* f_O2 .* Ox(:,i-1) *accel;            %OM oxidation
    R2a = Rc1 .* f_NO3 .* NO3(:,i-1);          %OM oxidation by NO3 
    R3a = Rc1 .* f_FeOH3 .* FeOH3(:,i-1) ;         %OM oxidation by Fe(OH)3
    R4a = Rc1 .* f_FeOOH .* FeOOH(:,i-1);          %OM oxidation by FeOOH
    R5a = Rc1 .* f_SO4 .* SO4(:,i-1) ;             %OM oxidation by SO4
    Ra = R1a+R2a+R3a+R4a+R5a;                      % degradation of OM1
    R1b = Rc2 .* f_O2 .* Ox(:,i-1) *accel;            %OM oxidation
    R2b = Rc2 .* f_NO3 .* NO3(:,i-1);          %OM oxidation by NO3 
    R3b = Rc2 .* f_FeOH3 .* FeOH3(:,i-1) ;         %OM oxidation by Fe(OH)3
    R4b = Rc2 .* f_FeOOH .* FeOOH(:,i-1);          %OM oxidation by FeOOH
    R5b = Rc2 .* f_SO4 .* SO4(:,i-1) ;             %OM oxidation by SO4
    Rb = R1b+R2b+R3b+R4b+R5b;                      % degradation of OM2

    R16a = k_pdesorb_a * FeOH3(:,i-1);
    R17a = k_pdesorb_b * FeOOH(:,i-1);

    % TODO: How to deal with phosphorus in both models in Phosporus part?
    % Baseline model part
    R18a = k_pdesorb_c .* AlOH3(:,i-1); % TODO: find out K_alum in R18a
    R_eq_PO4(:,i) = - R18a - R16a - R17a;

    R7 = k_tS_Fe * FeOH3(:,i-1) .* Sum_H2S;
    R3a = Rc1 .* f_FeOH3 .* FeOH3(:,i-1);          %OM oxidation by Fe(OH)3
    R3b = Rc2 .* f_FeOH3 .* FeOH3(:,i-1);          %OM oxidation by Fe(OH)3
    R3 = R3a+R3b; 
    R16b = f_pfe .* (4 * R3 + 2 * R7) ;
    R4a = Rc1 .* f_FeOOH .* FeOOH(:,i-1);          %OM oxidation by Fe(OH)3
    R4b = Rc2 .* f_FeOOH .* FeOOH(:,i-1);          %OM oxidation by Fe(OH)3
    R4 = R4a+R4b; 
    R17b = f_pfe .* (4 * R4);
    R19 = k_apa * (PO4(:,i-1) - kapa);
    R19 = (R19 >= 0) .* R19;
    PO4_dCbio(:,i) = bioirrigation(PO4(:,i-1), alfax, fi);
    R_eq_PO4_non_lin(:,i) = (P_index1 * Ra + P_index2 * Rb)*F + R16b + R17b - 2 * R19 + PO4_dCbio(:,i);


    % ============= S(0)====================================================================================
    % R_S0 = + R7 - R11 + R15b - R15a
    
    R7 = k_tS_Fe .* FeOH3(:,i-1) .* Sum_H2S; 
    R11 = k_FeS2pre .* FeS(:,i-1); 
    R15b = k_Sdis .* S8(:,i-1);
    R15a = k_Spre .* ones(n,1); % Should be matrix (n,1)
    R_eq_S0(:,i) =  - R11 - R15a;
    R_eq_S0_non_lin(:,i) = R7 + R15b + bioirrigation(S0(:,i-1), alfax, fi);

    % ============= S8 ====================================================================================
    % R_S8 = 4*R12 - R15b + R15a

    R15b = k_Sdis .* ones(n,1); % Should be matrix (n,1)
    R15a = k_Spre * S0(:,i-1);
    R12 = k_rhom * Ox(:,i-1) .* FeS(:,i-1);
    R_eq_S8(:,i) = - R15b;
    R_eq_S8_non_lin(:,i) = R15a + 4 * R12;

    % ============= FeS2====================================================================================
    % R_FeS2 = + R11 + R13 

    R11 = k_FeSpre .* FeS(:,i-1) .* S0(:,i-1);
    R13 = k_FeS2pre .* FeS(:,i-1) .* Sum_H2S;
    R_eq_FeS2(:,i) = 0;
    R_eq_FeS2_non_lin(:,i) = R13 + R11;

    % ============= Al(OH)3====================================================================================
    % R_AlOH3 = - R18a + R18b
    % TODO: add R18b

    R18a = k_pdesorb_c .* PO4(:,i-1);
    R_eq_AlOH3(:,i) = -R18a;
    R_eq_AlOH3_non_lin(:,i) = 0;

    % ============= PO4adsa ====================================================================================
    % R_PO4adsa = R16a - R16b 

    R3a = Rc1 .* f_FeOH3 .* FeOH3(:,i-1);          %OM oxidation by Fe(OH)3
    R3b = Rc2 .* f_FeOH3 .* FeOH3(:,i-1);          %OM oxidation by Fe(OH)3
    R3 = R3a+R3b; 
    R16b = f_pfe .* (4 * R3 + 2 * R7) ;
    R_eq_PO4adsa(:,i) = - R16b;

    R16a = k_pdesorb_a * FeOH3(:,i-1) .* PO4(:,i-1);
    R_eq_PO4adsa_non_lin(:,i) = R16a ;

      % ============= PO4adsb ====================================================================================
    % R_PO4adsb = R17a - R17b

    R4a = Rc1 .* f_FeOOH .* FeOOH(:,i-1);          %OM oxidation by FeOOH
    R4b = Rc2 .* f_FeOOH .* FeOOH(:,i-1);          %OM oxidation by FeOOH
    R4 = R4a+R4b; 
    R17b = f_pfe .* (4 * R4);
    R_eq_PO4adsb(:,i) = - R17b;

    R17a = k_pdesorb_b * FeOOH(:,i-1) .* PO4(:,i-1);
    R_eq_PO4adsb_non_lin(:,i) = R17a ;


    % ============= Ca2+ =========================================================================================
    % R_Ca2+ = -3*R19

    R19 = k_apa * (PO4(:,i-1) - kapa);
    R19 = (R19 >= 0) .* R19;
    R_eq_Ca2(:,i) = 0;
    R_eq_Ca2_non_lin(:,i) = -3*R19 + bioirrigation(Ca2(:,i-1), alfax, fi);
    

    % ============= Ca3PO42 ======================================================================================
    % R_Ca(PO4)2 = R19

    R19 = k_apa * (PO4(:,i-1) - kapa);
    R19 = (R19 >= 0) .* R19;
    R_eq_Ca3PO42(:,i) = 0;
    R_eq_Ca3PO42_non_lin(:,i) = R19;

    % ============= OMS ======================================================================================
    % R_OMS = R10

    R10 = k_oms * Sum_H2S .* (OM(:,i-1) + OMb(:,i-1));
    R_eq_OMS(:,i) = 0;
    R_eq_OMS_non_lin(:,i) = R10;


    % ============= H2CO3 ======================================================================================
    % R_H2CO3 = 0
    R_eq_H2CO3(:,i) = 0;
    R_eq_H2CO3_non_lin(:,i) = bioirrigation(H2CO3(:,i-1), alfax, fi);

    % ============= CO2 ======================================================================================
    % R_CO2 = ((1-y/x+2z/x)*R1 + (0.2-y/x+2z/x)R2 - (7+y/x-2z/x)R3 - (7+y/x-2z/x)R4 - (y/x-2z/x)R5)*F + 2R8 + 2R9
    R_eq_CO2(:,i) = 0;

    % In R_eq_non_lin should be all variables due to it is outside matrix multiplication
    R1a = Rc1 .* f_O2 .* Ox(:,i-1) *accel;            %OM oxidation
    R2a = Rc1 .* f_NO3 .* NO3(:,i-1);          %OM oxidation by NO3 
    R3a = Rc1 .* f_FeOH3 .* FeOH3(:,i-1) ;         %OM oxidation by Fe(OH)3
    R4a = Rc1 .* f_FeOOH .* FeOOH(:,i-1);          %OM oxidation by FeOOH
    R5a = Rc1 .* f_SO4 .* SO4(:,i-1) ;             %OM oxidation by SO4

    R1b = Rc2 .* f_O2 .* Ox(:,i-1) *accel;            %OM oxidation
    R2b = Rc2 .* f_NO3 .* NO3(:,i-1);          %OM oxidation by NO3 
    R3b = Rc2 .* f_FeOH3 .* FeOH3(:,i-1) ;         %OM oxidation by Fe(OH)3
    R4b = Rc2 .* f_FeOOH .* FeOOH(:,i-1);          %OM oxidation by FeOOH
    R5b = Rc2 .* f_SO4 .* SO4(:,i-1) ;             %OM oxidation by SO4

    R8 = k_Feox .* Fe2(:,i-1) .* Ox(:,i-1);
    R9 = k_amox * Ox(:,i-1) ./ (Km_oxao + Ox(:,i-1)) .* (NH4(:,i-1) ./ (Km_amao + NH4(:,i-1)));

    R_eq_CO2_non_lin(:,i) = ((1 - N_index1 + 2*P_index1)*R1a + (1 - N_index2 + 2*P_index2)*R1b + (0.2 - N_index1 + 2*P_index1)*R2a +  (0.2 - N_index2 + 2*P_index2)*R2b - (7 + N_index1 + 2*P_index1)*(R3a+R4a) - (7 + N_index2 + 2*P_index2)*(R3b+R4b) - (N_index1 - 2*P_index1)*R5a - (N_index2 - 2*P_index2)*R5b)*F  +  2*R8 + 2*R9 + bioirrigation(CO2(:,i-1), alfax, fi);

    % ============= HCO3(-) ======================================================================================
    % R_HCO3 = ((0.8 + y/x - 2z/x)*R2 + (8+y/x-2z/x)*(R3 + R4) + (1+y/x-2z/x)*R5)*F -2*R8 -2*R9
    R_eq_HCO3(:,i) = 0;
    R_eq_HCO3_non_lin(:,i) =((0.8 + N_index1 - 2*P_index1)*R2a + (0.8 + N_index2 - 2*P_index2)*R2b + (8+N_index1-2*P_index1)*(R3a + R4a) +(8+N_index2-2*P_index2)*(R3b + R4b) + (1+N_index1-2*P_index1)*R5a + (1+N_index2-2*P_index2)*R5b)*F -  2*R8 - 2*R9 + bioirrigation(HCO3(:,i-1), alfax, fi);

    % ============= CO3(2-) ======================================================================================
    % R_CO3 = 0
    R_eq_CO3(:,i) = 0;
    R_eq_CO3_non_lin(:,i) = bioirrigation(CO3(:,i-1), alfax, fi);

    % ============= H2S ======================================================================================
    % R_H2S = 0
    R_eq_H2S(:,i) = 0;
    R_eq_H2S_non_lin(:,i) = bioirrigation(H2S(:,i-1), alfax, fi);

    % ============= HS- ====================================================================================
    % R_HS = 0.5*R5*F - R6 - R7 + R14b - R14a - R10 -R13

    R10 = k_oms * (OM(:,i-1) + OMb(:,i-1));
    R_eq_HS(:,i) = -R10;

    R5b = Rc2 .* f_SO4 .* SO4(:,i-1);
    R5a = Rc1 .* f_SO4 .* SO4(:,i-1); 
    R5 = R5a + R5b;
    Sat_FeS(:,i-1) = Fe2(:,i-1) .* Sum_H2S ./ (H(:,i-1).^2 .* Ks_FeS);
    R14a = k_Fe_pre .* ( Sat_FeS(:,i-1) - 1);
    R14b  = k_Fe_dis .* FeS(:,i-1) .* ( 1 - Sat_FeS(:,i-1));
    R14a = (R14a >= 0) .* R14a; % can  be only non negative
    R14b  = (R14b >= 0) .* R14b; % can  be only non negative
    R6 = k_tsox * Ox(:,i-1) .* Sum_H2S;
    R7 = k_tS_Fe * FeOH3(:,i-1) .*  Sum_H2S;
    R13 = k_FeS2pre .* FeS(:,i-1) .* Sum_H2S;

    R_eq_HS_non_lin(:,i) = R5*F - R6 - R7 + R14b - R14a - R13 + bioirrigation(HS(:,i-1), alfax, fi); % NOTE: SatFeS have C(HS)

    % ============= NH3 ======================================================================================
    % R_NH3 = 0
    R_eq_NH3(:,i) = 0;
    R_eq_NH3_non_lin(:,i) = bioirrigation(NH3(:,i-1), alfax, fi);

    % ============= NH4 =====================================================================================
    % R_NH4 = y/x*(R1 + R2 + R3 + R4 + R5)*F - R9

    % All variables here are without C(NH4) in numerator due to it is will be added by matrix multiplication
    R9 = k_amox * Ox(:,i-1) ./ (Km_oxao + Ox(:,i-1)) .* (1 ./ (Km_amao + NH4(:,i-1))); %ammonia oxidation with MM kinetics AML
    R_eq_NH4(:,i) = - R9; 
    % In R_eq_non_lin should be all variables due to it is outside matrix multiplication
    R1a = Rc1 .* f_O2 .* Ox(:,i-1) *accel;            %OM oxidation
    R2a = Rc1 .* f_NO3 .* NO3(:,i-1);          %OM oxidation by NO3 
    R3a = Rc1 .* f_FeOH3 .* FeOH3(:,i-1) ;         %OM oxidation by Fe(OH)3
    R4a = Rc1 .* f_FeOOH .* FeOOH(:,i-1);          %OM oxidation by FeOOH
    R5a = Rc1 .* f_SO4 .* SO4(:,i-1) ;             %OM oxidation by SO4

    Ra = R1a+R2a+R3a+R4a+R5a;                      % degradation of OM1
    R1b = Rc2 .* f_O2 .* Ox(:,i-1) *accel;            %OM oxidation
    R2b = Rc2 .* f_NO3 .* NO3(:,i-1);          %OM oxidation by NO3 
    R3b = Rc2 .* f_FeOH3 .* FeOH3(:,i-1) ;         %OM oxidation by Fe(OH)3
    R4b = Rc2 .* f_FeOOH .* FeOOH(:,i-1);          %OM oxidation by FeOOH
    R5b = Rc2 .* f_SO4 .* SO4(:,i-1) ;             %OM oxidation by SO4
    Rb = R1b+R2b+R3b+R4b+R5b;                      % degradation of OM2
    R_eq_NH4_non_lin(:,i) = (N_index1 * Ra + N_index2 * Rb) * F + bioirrigation(NH4(:,i-1), alfax, fi);

    % ============= H(+) ======================================================================================
    % R_H = -0.8*R2*F + R6 -5*R7 - R14b + R14a + 3R18a - 3R18b + 3R16a + 3R17a
    R2a = Rc1 .* f_NO3 .* NO3(:,i-1);          %OM oxidation by NO3 
    R2b = Rc2 .* f_NO3 .* NO3(:,i-1);          %OM oxidation by NO3 
    R2  = (R2a+R2b);
    R6  = k_tsox * Ox(:,i-1) .* Sum_H2S;
    R7  = k_tS_Fe * FeOH3(:,i-1) .*  Sum_H2S;
    R14a = k_Fe_pre .* ( Sat_FeS(:,i-1) - 1);
    R14b  = k_Fe_dis .* FeS(:,i-1) .* ( 1 - Sat_FeS(:,i-1));
    R14a = (R14a >= 0) .* R14a; % can  be only non negative
    R14b  = (R14b >= 0) .* R14b; % can  be only non negative
    % TODO: add R18b
    R18a = k_pdesorb_c .* PO4(:,i-1) .* AlOH3(:,i-1);
    R_eq_H(:,i) = 0;
    R_eq_H_non_lin(:,i) = -0.8*R2*F + R6 - 5*R7 - R14b + R14a +3*R18a + bioirrigation(H(:,i-1), alfax, fi);

    % ============= OH(-) ======================================================================================
    % R_OH = 0
    R_eq_OH(:,i) = 0;
    R_eq_OH_non_lin(:,i) = bioirrigation(OH(:,i-1), alfax, fi);



    % =======================================================================================================
    % Updating matrices and Solving eq-s
    % =======================================================================================================
    if species_sediment('Oxygen') == true
      [LU_ox, RK_ox, BC_Ox_matrix] = update_matrices_solute( LU_ox0, RK_ox0, BC_Ox_matrix, ...
      BC_O(1,i-1), BC_O(1,i),R_eq_oxygen(2:end,i), gama, RA_ox0, RD_ox0, LA_ox0, LD_ox0, fi, dt);
      Ox(2:end,i)  = solving_eq(LU_ox,RK_ox,fi(2:end).*R_eq_Ox_non_lin(2:end,i)*dt,BC_Ox_matrix, Ox(2:end,i-1), fi(2:end));
    end

    if species_sediment('OM1') == true
      [LU_om, RK_om, BC_OM_matrix] = update_matrices_solid( LU_om0, RK_om0, BC_OM_matrix, ...
      R_eq_OM(:,i), gama, RA_om0, RD_om0, LA_om0, LD_om0, dt, dx, Db, fi, F_OM_top(i)/F, v);
      OM(:,i)      = solving_eq(LU_om,RK_om,(1-fi).*R_eq_OM_non_lin(:,i)*dt,BC_OM_matrix, OM(:,i-1), fi);
    end

    if species_sediment('OM2') == true
      [LU_omb, RK_omb, BC_OMb_matrix] = update_matrices_solid( LU_omb0, RK_omb0, BC_OMb_matrix, ...
      R_eq_OMb(:,i), gama, RA_omb0, RD_omb0, LA_omb0, LD_omb0, dt, dx, Db, fi, F_OMb_top(i)/F, v);
      OMb(:,i)     = solving_eq(LU_omb,RK_omb,(1-fi).*R_eq_OMb_non_lin(:,i)*dt,BC_OMb_matrix, OMb(:,i-1), fi);
    end

    if species_sediment('NO3') == true
      [LU_NO3, RK_NO3, BC_NO3_matrix] = update_matrices_solute( LU_NO30, RK_NO30, BC_NO3_matrix, ...
      BC_NO3(1,i-1), BC_NO3(1,i),R_eq_NO3(2:end,i), gama, RA_NO30, RD_NO30, LA_NO30, LD_NO30, fi, dt);
      NO3(2:end,i) = solving_eq(LU_NO3,RK_NO3,fi(2:end).*R_eq_NO3_non_lin(2:end,i)*dt,BC_NO3_matrix, NO3(2:end,i-1), fi(2:end));
    end

    if species_sediment('FeOH3') == true
      [LU_FeOH3, RK_FeOH3, BC_FeOH3_matrix] = update_matrices_solid( LU_FeOH30, RK_FeOH30, BC_FeOH3_matrix, ...
      R_eq_FeOH3(:,i), gama, RA_FeOH30, RD_FeOH30, LA_FeOH30, LD_FeOH30, dt, dx, Db, fi, F_FeOH3_top(i)/F, v);
      FeOH3(:,i)   = solving_eq(LU_FeOH3,RK_FeOH3,(1-fi).*R_eq_FeOH3_non_lin(:,i)*dt,BC_FeOH3_matrix, FeOH3(:,i-1), fi);
    end

    if species_sediment('SO4') == true
      [LU_SO4, RK_SO4, BC_SO4_matrix] = update_matrices_solute( LU_SO40, RK_SO40, BC_SO4_matrix, ...
      BC_SO4(1,i-1), BC_SO4(1,i),R_eq_SO4(2:end,i), gama, RA_SO40, RD_SO40, LA_SO40, LD_SO40, fi, dt);
      SO4(2:end,i) = solving_eq(LU_SO4,RK_SO4,fi(2:end).*R_eq_SO4_non_lin(2:end,i)*dt,BC_SO4_matrix, SO4(2:end,i-1), fi(2:end));
    end

    if species_sediment('NH4') == true
      [LU_NH4, RK_NH4, BC_NH4_matrix] = update_matrices_solute( LU_NH40, RK_NH40, BC_NH4_matrix, ...
      BC_NH4(1,i-1), BC_NH4(1,i),R_eq_NH4(2:end,i), gama, RA_NH40, RD_NH40, LA_NH40, LD_NH40, fi, dt);
      NH4(2:end,i) = solving_eq(LU_NH4,RK_NH4,fi(2:end).*R_eq_NH4_non_lin(2:end,i)*dt,BC_NH4_matrix, NH4(2:end,i-1), fi(2:end));
    end

    if species_sediment('Fe2') == true
      [LU_Fe2, RK_Fe2, BC_Fe2_matrix] = update_matrices_solute( LU_Fe20, RK_Fe20, BC_Fe2_matrix, ...
      BC_Fe2(1,i-1), BC_Fe2(1,i),R_eq_Fe2(2:end,i), gama, RA_Fe20, RD_Fe20, LA_Fe20, LD_Fe20, fi, dt);
      Fe2(2:end,i) = solving_eq(LU_Fe2,RK_Fe2,fi(2:end).*R_eq_Fe2_non_lin(2:end,i)*dt,BC_Fe2_matrix, Fe2(2:end,i-1), fi(2:end));
    end

    if species_sediment('FeOOH') == true
      [LU_FeOOH, RK_FeOOH, BC_FeOOH_matrix] = update_matrices_solid( LU_FeOOH0, RK_FeOOH0, BC_FeOOH_matrix, ...
      R_eq_FeOOH(:,i), gama, RA_FeOOH0, RD_FeOOH0, LA_FeOOH0, LD_FeOOH0, dt, dx, Db, fi, F_FeOOH_top(i)/F, v);
      FeOOH(:,i)   = solving_eq(LU_FeOOH,RK_FeOOH,(1-fi).*R_eq_FeOOH_non_lin(:,i)*dt,BC_FeOOH_matrix, FeOOH(:,i-1), fi);
    end

    if species_sediment('H2S') == true
      [LU_H2S, RK_H2S, BC_H2S_matrix] = update_matrices_solute( LU_H2S0, RK_H2S0, BC_H2S_matrix, ...
      BC_H2S(1,i-1), BC_H2S(1,i),R_eq_H2S(2:end,i), gama, RA_H2S0, RD_H2S0, LA_H2S0, LD_H2S0, fi, dt);
      H2S(2:end,i) = solving_eq(LU_H2S, RK_H2S, fi(2:end).*R_eq_H2S_non_lin(2:end,i)*dt, BC_H2S_matrix,  H2S(2:end,i-1), fi(2:end));
    end

    if species_sediment('HS') == true
      [LU_HS, RK_HS, BC_HS_matrix] = update_matrices_solute( LU_HS0, RK_HS0, BC_HS_matrix, ...
      BC_HS(1,i-1), BC_HS(1,i),R_eq_HS(2:end,i), gama, RA_HS0, RD_HS0, LA_HS0, LD_HS0, fi, dt);
      HS(2:end,i)  = solving_eq(LU_HS, RK_HS, fi(2:end).*R_eq_HS_non_lin(2:end,i)*dt, BC_HS_matrix,  HS(2:end,i-1), fi(2:end));
    end

    if species_sediment('FeS') == true
      [LU_FeS, RK_FeS, BC_FeS_matrix] = update_matrices_solid( LU_FeS0, RK_FeS0, BC_FeS_matrix, ...
      R_eq_FeS(:,i), gama, RA_FeS0, RD_FeS0, LA_FeS0, LD_FeS0, dt, dx, Db, fi, F_FeS_top(i)/F, v);
      FeS(:,i)     = solving_eq(LU_FeS,RK_FeS,(1-fi).*R_eq_FeS_non_lin(:,i)*dt,BC_FeS_matrix, FeS(:,i-1), fi);
    end

    if species_sediment('S0') == true
      [LU_S0, RK_S0, BC_S0_matrix] = update_matrices_solute( LU_S00, RK_S00, BC_S0_matrix, ...
      BC_S0(1,i-1), BC_S0(1,i),R_eq_S0(2:end,i), gama, RA_S00, RD_S00, LA_S00, LD_S00, fi, dt);
      S0(2:end,i)  = solving_eq(LU_S0, RK_S0, fi(2:end).*R_eq_S0_non_lin(2:end,i)*dt, BC_S0_matrix,  S0(2:end,i-1), fi(2:end));
    end

    if species_sediment('PO4') == true
      [LU_PO4, RK_PO4, BC_PO4_matrix] = update_matrices_solute( LU_PO40, RK_PO40, BC_PO4_matrix, ...
      BC_PO4(1,i-1), BC_PO4(1,i),R_eq_PO4(2:end,i), gama, RA_PO40, RD_PO40, LA_PO40, LD_PO40, fi, dt);
      PO4(2:end,i) = solving_eq(LU_PO4, RK_PO4, fi(2:end).*R_eq_PO4_non_lin(2:end,i)*dt, BC_PO4_matrix,  PO4(2:end,i-1), fi(2:end));
    end
    
    if species_sediment('S8') == true
      [LU_S8, RK_S8, BC_S8_matrix] = update_matrices_solid( LU_S80, RK_S80, BC_S8_matrix, ...
      R_eq_S8(:,i), gama, RA_S80, RD_S80, LA_S80, LD_S80, dt, dx, Db, fi, F_S8_top(i)/F, v);
      S8(:,i)      = solving_eq(LU_S8,RK_S8,(1-fi).*R_eq_S8_non_lin(:,i)*dt,BC_S8_matrix, S8(:,i-1), fi);
    end

    if species_sediment('FeS2') == true
      [LU_FeS2, RK_FeS2, BC_FeS2_matrix] = update_matrices_solid( LU_FeS20, RK_FeS20, BC_FeS2_matrix, ...
      R_eq_FeS2(:,i), gama, RA_FeS20, RD_FeS20, LA_FeS20, LD_FeS20, dt, dx, Db, fi, F_FeS2_top(i)/F, v);
      FeS2(:,i)    = solving_eq(LU_FeS2,RK_FeS2,(1-fi).*R_eq_FeS2_non_lin(:,i)*dt,BC_FeS2_matrix, FeS2(:,i-1), fi);
    end

    if species_sediment('AlOH3') == true
      [LU_AlOH3, RK_alum, BC_AlOH3_matrix] = update_matrices_solid( LU_AlOH30, RK_alum0, BC_AlOH3_matrix, ...
      R_eq_AlOH3(:,i), gama, RA_AlOH30, RD_AlOH30, LA_AlOH30, LD_AlOH30, dt, dx, Db, fi, F_AlOH3_top(i)/F, v);
      AlOH3(:,i)   = solving_eq(LU_AlOH3,RK_alum,(1-fi).*R_eq_AlOH3_non_lin(:,i)*dt,BC_AlOH3_matrix, AlOH3(:,i-1), fi);
    end

    if species_sediment('PO4adsa') == true
      [LU_PO4adsa, RK_PO4adsa, BC_PO4adsa_matrix] = update_matrices_solid( LU_PO4adsa0, RK_PO4adsa0, BC_PO4adsa_matrix, ...
      R_eq_PO4adsa(:,i), gama, RA_PO4adsa0, RD_PO4adsa0, LA_PO4adsa0, LD_PO4adsa0, dt, dx, Db, fi, F_PO4adsa_top(i)/F, v);
      PO4adsa(:,i) = solving_eq(LU_PO4adsa,RK_PO4adsa,(1-fi).*R_eq_PO4adsa_non_lin(:,i)*dt,BC_PO4adsa_matrix, PO4adsa(:,i-1), fi);
    end

    if species_sediment('PO4adsb') == true
      [LU_PO4adsb, RK_PO4adsb, BC_PO4adsb_matrix] = update_matrices_solid( LU_PO4adsb0, RK_PO4adsb0, BC_PO4adsb_matrix, ...
      R_eq_PO4adsb(:,i), gama, RA_PO4adsb0, RD_PO4adsb0, LA_PO4adsb0, LD_PO4adsb0, dt, dx, Db, fi, F_PO4adsb_top(i)/F, v);
      PO4adsb(:,i) = solving_eq(LU_PO4adsb,RK_PO4adsb,(1-fi).*R_eq_PO4adsb_non_lin(:,i)*dt,BC_PO4adsb_matrix, PO4adsb(:,i-1), fi);
    end

    if species_sediment('Ca2') == true
      [LU_Ca2, RK_Ca2, BC_Ca2_matrix] = update_matrices_solute( LU_Ca20, RK_Ca20, BC_Ca2_matrix, ...
      BC_Ca2(1,i-1), BC_Ca2(1,i),R_eq_Ca2(2:end,i), gama, RA_Ca20, RD_Ca20, LA_Ca20, LD_Ca20, fi, dt);
      Ca2(2:end,i) = solving_eq(LU_Ca2,RK_Ca2,fi(2:end).*R_eq_Ca2_non_lin(2:end,i)*dt,BC_Ca2_matrix, Ca2(2:end,i-1), fi(2:end));
    end

    if species_sediment('Ca3PO42') == true
      [LU_Ca3PO42, RK_Ca3PO42, BC_Ca3PO42_matrix] = update_matrices_solid( LU_Ca3PO420, RK_Ca3PO420, BC_Ca3PO42_matrix, ...
      R_eq_Ca3PO42(:,i), gama, RA_Ca3PO420, RD_Ca3PO420, LA_Ca3PO420, LD_Ca3PO420, dt, dx, Db, fi, F_Ca3PO42_top(i)/F, v);
      Ca3PO42(:,i) = solving_eq(LU_Ca3PO42,RK_Ca3PO42,(1-fi).*R_eq_Ca3PO42_non_lin(:,i)*dt,BC_Ca3PO42_matrix, Ca3PO42(:,i-1), fi);
    end

    if species_sediment('OMS') == true
      [LU_OMS, RK_OMS, BC_OMS_matrix] = update_matrices_solid( LU_OMS0, RK_OMS0, BC_OMS_matrix, ...
      R_eq_OMS(:,i), gama, RA_OMS0, RD_OMS0, LA_OMS0, LD_OMS0, dt, dx, Db, fi, F_OMS_top(i)/F, v);
      OMS(:,i)     = solving_eq(LU_OMS,RK_OMS,(1-fi).*R_eq_OMS_non_lin(:,i)*dt,BC_OMS_matrix, OMS(:,i-1), fi);
    end

    if species_sediment('H') == true
      [LU_H, RK_H, BC_H_matrix] = update_matrices_solute( LU_H0, RK_H0, BC_H_matrix, ...
      BC_H(1,i-1), BC_H(1,i),R_eq_H(2:end,i), gama, RA_H0, RD_H0, LA_H0, LD_H0, fi, dt);
      H(2:end,i) = solving_eq(LU_H,RK_H,fi(2:end).*R_eq_H_non_lin(2:end,i)*dt,BC_H_matrix, H(2:end,i-1), fi(2:end));
    end

    if species_sediment('OH') == true
      [LU_OH, RK_OH, BC_OH_matrix] = update_matrices_solute( LU_OH0, RK_OH0, BC_OH_matrix, ...
      BC_OH(1,i-1), BC_OH(1,i),R_eq_OH(2:end,i), gama, RA_OH0, RD_OH0, LA_OH0, LD_OH0, fi, dt);
      OH(2:end,i) = solving_eq(LU_OH,RK_OH,fi(2:end).*R_eq_OH_non_lin(2:end,i)*dt,BC_OH_matrix, OH(2:end,i-1), fi(2:end));
    end

    if species_sediment('CO2') == true
      [LU_CO2, RK_CO2, BC_CO2_matrix] = update_matrices_solute( LU_CO20, RK_CO20, BC_CO2_matrix, ...
      BC_CO2(1,i-1), BC_CO2(1,i),R_eq_CO2(2:end,i), gama, RA_CO20, RD_CO20, LA_CO20, LD_CO20, fi, dt);
      CO2(2:end,i) = solving_eq(LU_CO2,RK_CO2,fi(2:end).*R_eq_CO2_non_lin(2:end,i)*dt,BC_CO2_matrix, CO2(2:end,i-1), fi(2:end));
    end

    if species_sediment('CO3') == true
      [LU_CO3, RK_CO3, BC_CO3_matrix] = update_matrices_solute( LU_CO30, RK_CO30, BC_CO3_matrix, ...
      BC_CO3(1,i-1), BC_CO3(1,i),R_eq_CO3(2:end,i), gama, RA_CO30, RD_CO30, LA_CO30, LD_CO30, fi, dt);
      CO3(2:end,i) = solving_eq(LU_CO3,RK_CO3,fi(2:end).*R_eq_CO3_non_lin(2:end,i)*dt,BC_CO3_matrix, CO3(2:end,i-1), fi(2:end));
    end

    if species_sediment('HCO3') == true
      [LU_HCO3, RK_HCO3, BC_HCO3_matrix] = update_matrices_solute( LU_HCO30, RK_HCO30, BC_HCO3_matrix, ...
      BC_HCO3(1,i-1), BC_HCO3(1,i),R_eq_HCO3(2:end,i), gama, RA_HCO30, RD_HCO30, LA_HCO30, LD_HCO30, fi, dt);
      HCO3(2:end,i) = solving_eq(LU_HCO3,RK_HCO3,fi(2:end).*R_eq_HCO3_non_lin(2:end,i)*dt,BC_HCO3_matrix, HCO3(2:end,i-1), fi(2:end));
    end

    if species_sediment('NH3') == true
      [LU_NH3, RK_NH3, BC_NH3_matrix] = update_matrices_solute( LU_NH30, RK_NH30, BC_NH3_matrix, ...
      BC_NH3(1,i-1), BC_NH3(1,i),R_eq_NH3(2:end,i), gama, RA_NH30, RD_NH30, LA_NH30, LD_NH30, fi, dt);
      NH3(2:end,i) = solving_eq(LU_NH3,RK_NH3,fi(2:end).*R_eq_NH3_non_lin(2:end,i)*dt,BC_NH3_matrix, NH3(2:end,i-1), fi(2:end));
    end

    if species_sediment('H2CO3') == true
      [LU_H2CO3, RK_H2CO3, BC_H2CO3_matrix] = update_matrices_solute( LU_H2CO30, RK_H2CO30, BC_H2CO3_matrix, ...
      BC_H2CO3(1,i-1), BC_H2CO3(1,i),R_eq_H2CO3(2:end,i), gama, RA_H2CO30, RD_H2CO30, LA_H2CO30, LD_H2CO30, fi, dt);
      H2CO3(2:end,i) = solving_eq(LU_H2CO3,RK_H2CO3,fi(2:end).*R_eq_H2CO3_non_lin(2:end,i)*dt,BC_H2CO3_matrix, H2CO3(2:end,i-1), fi(2:end));
    end



    % Add new species before this line.
    % =======================================================================================================


    % pH Module
    if sediment_params('pH') == true
      [H(:,i), OH(:,i), H2CO3(:,i), HCO3(:,i), CO2(:,i), CO3(:,i), NH3(:,i), NH4(:,i), HS(:,i), H2S(:,i)] = pH_module(sediment_params('pH algorithm'), H(:,i), OH(:,i), H2CO3(:,i), HCO3(:,i), CO2(:,i), CO3(:,i), NH3(:,i), NH4(:,i), HS(:,i), H2S(:,i), Fe2(:,i), Ca2(:,i), NO3(:,i), SO4(:,i), PO4(:,i));
    end


    % Collecting R integrated values of OM oxidations by TEA 
    Rc1 = K_OM.*OM(:,i-1);
    Rc2 = K_OMb.*OMb(:,i-1);

    R1a = Rc1 .* f_O2 .* Ox(:,i-1) *accel;
    R1b = Rc2 .* f_O2 .* Ox(:,i-1) *accel;
    R1_graphs(:,i-1) = R1a+R1b;

    R2b = Rc2 .* f_NO3 .* NO3(:,i-1);
    R2a = Rc1 .* f_NO3 .* NO3(:,i-1);
    R2_graphs(:,i-1) = R2a+R2b;


    R3a = Rc1 .* f_FeOH3 .* FeOH3(:,i-1);
    R3b = Rc2 .* f_FeOH3 .* FeOH3(:,i-1);
    R3_graphs(:,i-1)  = R3a+R3b;

    
    R4a = Rc1 .* f_FeOOH .* FeOOH(:,i-1);
    R4b = Rc2 .* f_FeOOH .* FeOOH(:,i-1);
    R4_graphs(:,i-1)  = R4a+R4b;


    R5a = Rc1 .* f_SO4 .* SO4(:,i-1) ;
    R5b = Rc2 .* f_SO4 .* SO4(:,i-1);
    R5_graphs(:,i-1)  = R5a+R5b;


    R6_graphs(:,i-1) = k_tsox * Ox(:,i-1) .* Sum_H2S;


  end



  sediment_concentrations = {...
    Ox(:,end),      'Oxygen';
    OM(:,end),      'OM1';
    OMb(:,end),     'OM2';
    NO3(:,end),     'NO3';
    FeOH3(:,end),   'FeOH3';
    SO4(:,end),     'SO4';
    NH4(:,end),     'NH4';
    Fe2(:,end),     'Fe2';
    FeOOH(:,end),   'FeOOH';
    H2S(:,end),     'H2S';
    HS(:,end),      'HS';
    FeS(:,end),     'FeS';
    S0(:,end),      'S0';
    PO4(:,end),     'PO4';
    S8(:,end),      'S8';
    FeS2(:,end),    'FeS2';
    AlOH3(:,end),   'AlOH3';
    PO4adsa(:,end), 'PO4adsa';
    PO4adsb(:,end), 'PO4adsb';
    Ca2(:,end),     'Ca2';
    Ca3PO42(:,end), 'Ca3PO42';
    OMS(:,end),     'OMS';
    H(:,end),       'H';
    OH(:,end),      'OH';
    CO2(:,end),     'CO2';
    CO3(:,end),     'CO3';
    HCO3(:,end),    'HCO3';
    NH3(:,end),     'NH3';
    H2CO3(:,end),   'H2CO3';
    };
    sediment_concentrations = containers.Map({sediment_concentrations{:,2}},{sediment_concentrations{:,1}});  

  R_values = {     
    daily_average(R1_graphs, m) ,              'R1';
    integrate_over_depth(R1_graphs, dx, m),    'R1 integrated';
    daily_average(R2_graphs, m) ,              'R2';
    integrate_over_depth(R2_graphs, dx, m),    'R2 integrated';
    daily_average(R3_graphs,m),                'R3';
    integrate_over_depth(R3_graphs, dx, m),    'R3 integrated';
    daily_average(R4_graphs,m),                'R4';
    integrate_over_depth(R4_graphs, dx, m),    'R4 integrated';
    daily_average(R5_graphs, m),               'R5';
    integrate_over_depth(R5_graphs, dx, m),    'R5 integrated';
    daily_average(R6_graphs,m),                'R6'; 
    integrate_over_depth(R6_graphs, dx, m),    'R6 integrated'; 
    };

  % Estimating of the water-column and sediment interaction due to diffusion and bioirrigation:
  Ox_flux = top_sediment_diffusion_flux(Ox(:,end), D_O2, 32000, dx, fi); %  + sum(Ox_dCbio(:,end)) * dx;
  PO4_flux = top_sediment_diffusion_flux(PO4(:,end),D_PO4, 94971, dx, fi); %  + sum(PO4_dCbio(:,end)) * dx;
  NO3_flux = top_sediment_diffusion_flux(NO3(:,end),D_NO3, 62004, dx, fi); %  + sum(bioirrigation(NO3(:,i-1), alfax, fi)) * dx;

  sediment_diffusion_fluxes = {...
    Ox_flux,                   'O2 flux';
    sediments_bc('OM1_fx'),    'OM1 flux';
    sediments_bc('OM2_fx'),    'OM2 flux';
    PO4_flux,                  'PO4 flux';
    NO3_flux,                  'NO3 flux';
    sediments_bc('FeOH3_fx'),  'FeOH3 flux';
  };

  % Flux due to bioirrigation:
  sediment_bioirrigation_fluxes = {...
    top_sediment_bioirrigation_flux(Ox_dCbio,32000,dx),          'Oxygen bioirrigation flux';
    top_sediment_bioirrigation_flux(PO4_dCbio,94971,dx),         'PO4 bioirrigation flux';
  };
  

  % Debugging purposes
 if any(isnan(PO4_flux)) | any(isnan(Ox_flux)) | any(isnan(Ox)) | any(isnan(OM)) | any(isnan(OMb)) | any(isnan(NO3)) | any(isnan(FeOH3)) | any(isnan(SO4)) | any(isnan(NH4)) | any(isnan(Fe2)) | any(isnan(FeOOH)) | any(isnan(H2S)) | any(isnan(HS)) | any(isnan(FeS)) | any(isnan(S0)) | any(isnan(PO4)) | any(isnan(S8)) | any(isnan(FeS2)) | any(isnan(AlOH3)) | any(isnan(PO4adsa)) | any(isnan(PO4adsb)) | any(isnan(H)) | any(isnan(Ca2)) | any(isnan(Ca3PO42)) | any(isnan(OMS)) | any(isnan(OH)) | any(isnan(HCO3)) | any(isnan(CO2)) | any(isnan(CO3)) | any(isnan(NH3)) | any(isnan(H2CO3))
%fprintf('The following have NaN values: PO4_flux %d, Ox_flux %d, Ox %d, OM %d, OMb %d, OMb %d, NO3 %d, FeOH3 %d, SO4 %d, NH4 %d, Fe2 %d, FeOOH %d, H2S %d, HS %d, FeS %d, SO %d, PO4 %d, S8 %d, FeS2 %d, AlOH3 % d, PO4adsa %d, Po4adsb %d, H %d, Ca2 %d, Ca3Po42 %d, OMS %d, OH %d, HCO3 %d, CO2 %d, CO3 %d, NH3 %d, H2CO3 %d \n', any(isnan(PO4_flux)) , any(isnan(Ox_flux)) , any(isnan(Ox)) , any(isnan(OM)) , any(isnan(OMb)) , any(isnan(NO3)) , any(isnan(FeOH3)) , any(isnan(SO4)) , any(isnan(NH4)) , any(isnan(Fe2)) , any(isnan(FeOOH)) , any(isnan(H2S)) , any(isnan(HS)) , any(isnan(FeS)) , any(isnan(S0)) , any(isnan(PO4)) , any(isnan(S8)) , any(isnan(FeS2)) , any(isnan(AlOH3)) , any(isnan(PO4adsa)) , any(isnan(PO4adsb)) , any(isnan(H)) , any(isnan(Ca2)) , any(isnan(Ca3PO42)) , any(isnan(OMS)) , any(isnan(OH)) , any(isnan(HCO3)) , any(isnan(CO2)) , any(isnan(CO3)) , any(isnan(NH3)) , any(isnan(H2CO3)));
   error('Breaking out of Sediments function: NaN values');
  end

end


%% top_sediment_bioirrigation_fluxes_flux: returns the flux of species at SWI converted to th units used in WC [ mg m-2 d-1 ].
function [flux] = top_sediment_bioirrigation_flux(dC, M_C, dx)
    % dC - changing in concentration due to bioirrigation [umol cm-3]
    % dC_wc - rate of bioirrigation [mg m-3 d-1]
    % M_C - molar mass in [ mg mol-1]
    % dx - mesh size [cm];


    dx_m=dx/100; %                    [cm] -> [m]
    dC_wc = sum(dC,2) .* M_C; %                [umol cm-3] -> [mg m-3 d-1], 
    % where sum of changing concentrations during the day is the rate per day due to MyLake time step is 1 day.
    flux = sum(dC_wc)*dx_m;
end

function [int_rate] = integrate_over_depth(R, dx, m)
  %% integrate_over_depth: integrates the rates of reaction over the depth and return the average values for the current day
  % R - rate of interest
  % dx - the mesh size
  % m - number of time step in 1 day
  int_rate = daily_average(sum(R,1)*dx, m);
end

%% daily_average: returns the average rate on a daily basis
function [averaged] = daily_average(R, m)
  % R - rate of interest
  % m - number of time step in 1 day
  averaged = sum(R,2)/(m-1);
end

function [flux] = top_sediment_diffusion_flux(C, D, M_C, dx, fi)
  % calculates flux of the particular dissolved specie through the top boundary of the sediments
  % in [ mg m-2 d-1 ] units
  % C(1) - BC of dissolved species
  % C - concentration
  % D - diffusion coefficient
  % M_C - molar mass in [ mg mol-1]
  % fi - porosity 
  
  dx_m=dx/100; %                    [cm] -> [m]
  dC = (C(2)- C(1)) * M_C; %       [umol cm-3] -> [mg m-3]
  D_m = D / 10^4 / 365; %           [cm2 y-1] -> [m2 d-1]
  flux = - D_m * fi(1) * dC / dx_m; %  [ mg m-2 d-1 ]
end

function bioC = bioirrigation(C, alfax, fi)
  % bioirrigation is the "artificial" function represents the bioirrigation by organisms in the sediment (worms etc) implemented according to Boudreau, B.P., 1999.
  % Co - bc wc-sediments value of current species
  % C - concentration profile of current species
  % fi - porosity 
  Co = C(1);
  bioC = - fi .* alfax .* (C - Co); 
end


function [H, OH, H2CO3, HCO3, CO2, CO3, NH3, NH4, HS, H2S] = pH_module(algorithm, H, OH, H2CO3, HCO3, CO2, CO3, NH3, NH4, HS, H2S, Fe2, Ca2, NO3, SO4, PO4)
  %% pH_module: pH equilibrium function 
  % 0. No pH module
  % 1. Stumm, W. & Morgan, J., 1995. Aquatic Chemistry. implemented in MATLAB
  % 2. Stumm, W. & Morgan, J., 1995. Aquatic Chemistry. implemented in C++
  % 3. Delta function (under construction)
   
  % NOTE: First point is boundary condition therefore start FOR loop from 2:end 
    
    if algorithm == 1 % Stumm, W. & Morgan, J., 1995. Aquatic Chemistry. implemented in MATLAB
      % The fasted algorithm is Levenberg-Marquardt
      options = optimoptions('fsolve','Algorithm','levenberg-marquardt','Display','off','TolFun',1e-7, 'TolX',1e-7); % Option to display output

      for i=1:size(H,1)   % Stumm, W. & Morgan, J., 1995. Aquatic Chemistry. implemented in MATLAB 
        % initial guess
        x = [sqrt(H(i)); sqrt(HCO3(i)); sqrt(CO2(i)); sqrt(CO3(i)); sqrt(NH3(i)); sqrt(NH4(i)); sqrt(HS(i)); sqrt(H2S(i)); sqrt(OH(i)); sqrt(H2CO3(i))];
        [x,fval] = fsolve(@SM_eqs, x, options, sqrt(H(i)), sqrt(OH(i)), sqrt(H2CO3(i)), sqrt(HCO3(i)), sqrt(CO2(i)), sqrt(CO3(i)), sqrt(NH3(i)), sqrt(NH4(i)), sqrt(HS(i)), sqrt(H2S(i)), sqrt(Fe2(i)), sqrt(Ca2(i)), sqrt(NO3(i)), sqrt(SO4(i)), sqrt(PO4(i))); % Call solver
        H(i) = x(1)^2;
        HCO3(i) = x(2)^2;
        CO2(i) = x(3)^2;
        CO3(i) = x(4)^2;
        NH3(i) = x(5)^2;
        NH4(i) = x(6)^2;
        HS(i) = x(7)^2;
        H2S(i) = x(8)^2;
        OH(i) = x(9)^2;
        H2CO3(i) = x(10)^2;
      end

    elseif algorithm == 2 % Stumm, W. & Morgan, J., 1995. Aquatic Chemistry. implemented in C++
      for i=1:size(H,1)
        in =[H(i) HCO3(i) CO2(i) CO3(i) NH3(i) NH4(i) HS(i) H2S(i) OH(i) H2CO3(i) Fe2(i) Ca2(i) NO3(i) SO4(i) PO4(i)];
        [out] = pH(in);
        H(i) = out(1);
        HCO3(i) = out(2);
        CO2(i) = out(3);
        CO3(i) = out(4);
        NH3(i) = out(5);
        NH4(i) = out(6);
        HS(i) = out(7);
        H2S(i) = out(8);
        OH(i) = out(9);
        H2CO3(i) = out(10);
      end

    elseif algorithm == 3 % Delta function
      options = optimoptions('fsolve','Algorithm','levenberg-marquardt','Display','off','TolFun',1e-7, 'TolX',1e-7); % Option to display output
      delta = [0,0,0,0,0,0,0,0,0];
      for i=1:size(H,1)
        [delta,fval] = fsolve(@delta_eqs, delta, options, H(i), OH(i), HCO3(i), CO2(i), CO3(i), NH3(i), NH4(i), HS(i), H2S(i)); % Call solver
        H(i) = H(i) + delta(1)+delta(2)+delta(3)+delta(4)+delta(5);
        OH(i) = OH(i) + delta(5);
        HCO3(i) = HCO3(i) + delta(1) - delta(2);
        CO3(i) = CO3(i) + delta(2);
        CO2(i) = CO2(i) - delta(1);
        NH3(i) = NH3(i) + delta(3);
        NH4(i) = NH4(i) - delta(3);
        HS(i) = HS(i) + delta(4);
        H2S(i) = H2S(i) - delta(4);
      end
    end
end


function F = SM_eqs(x, H, OH, H2CO3, HCO3, CO2, CO3, NH3, NH4, HS, H2S, Fe2, Ca2, NO3, SO4, PO4)
  %% Equations according to Stumm, W. & Morgan, J., 1995. Aquatic Chemistry implemented in MATLAB.

  % x(1)^2=H;   % x(2)^2=HCO3;  % x(3)^2=CO2;  % x(4)^2=CO3;  % x(5)^2=NH3;  % x(6)^2=NH4;  % x(7)^2=HS;  % x(8)^2=H2S;  % x(9)^2=OH;

  Kc1=5.01*10^(-7).*10^6; Kc2=4.78*10^(-11).*10^6; Knh=5.62*10^(-10).*10^6; Khs=1.3*10^(-7).*10^6; Kw=10^(-14).*10^12; Kc0 = 1.7*10^(-3); 

  F = [ 
      x(10)^2 - Kc0*x(3)^2;
      x(1)^2 * x(2)^2 - Kc1*x(10)^2;
      x(1)^2 * x(4)^2 - Kc2*x(2)^2;
      x(1)^2 * x(5)^2 - Knh*x(6)^2;
      x(1)^2 * x(7)^2 - Khs*x(8)^2;
      x(1)^2 * x(9)^2 - Kw;
     % mass balance
     x(5)^2 + x(6)^2 - NH4^2 - NH3^2;
     x(7)^2 + x(8)^2 - HS^2 - H2S^2;
     x(4)^2 + x(2)^2 + x(3)^2 + x(10)^2 - CO3^2 - HCO3^2 - CO2^2 - H2CO3^2;
     % charge balance
     x(1)^2 + x(6)^2 + 2*Fe2^2 + 2*Ca2^2 - (x(2)^2 + 2*x(4)^2 + x(7)^2 + x(9)^2 + NO3^2 + 2*SO4^2 + 3*PO4^2);
     ];
end

function F = delta_eqs(delta, H, OH, HCO3, CO2, CO3, NH3, NH4, HS, H2S)
  %% equations according to Delta function
  Kc1=10^(-6.4).*10^6; Kc2=10^(-10.3).*10^6; Knh=10^(-9.3).*10^6; Khs=10^(-7).*10^6; Kw=10^(-14).*10^12;
  
  F = [(H + delta(1) + delta(2) + delta(3) + delta(4) + delta(5) )* (HCO3 + delta(1) - delta(2) ) - Kc1*(CO2 - delta(1));
     (H + delta(1) + delta(2) + delta(3) + delta(4) + delta(5) )* (CO3 + delta(2) ) -  Kc2*(HCO3 - delta(2) + delta(1));
     (H + delta(1) + delta(2) + delta(3) + delta(4) + delta(5) )* (NH3 + delta(3) ) - Knh* (NH4 - delta(3));
     (H + delta(1) + delta(2) + delta(3) + delta(4) + delta(5) )* (HS + delta(4) ) - Khs* (H2S - delta(4));
     (H + delta(1) + delta(2) + delta(3) + delta(4) + delta(5) )* (OH + delta(5) ) - Kw;
     ];
end


function [ LU, RK, BC ] = update_matrices_solid( LU, RK, BC, R,gama,RA, RD, LA, LD, dt, dx, D, fi, F, v)
  %UPDATE_MATRICES function updates the matrices each time step according to the sources and sinks
  % and boundary conditions
  LR = dt * gama .* (1-fi);
  RR = dt * (1 - gama) .* (1-fi);

  LU(eye(size(LU))~=0) = diag(LU) - LR .* R;
  RK(eye(size(RK))~=0) = diag(RK) + RR .* R;
  
  BC(1) = 2*dx*F/(D*(1-fi(1)))*(LD(1)+LA(1)+RA(1)+RD(1));
end

function [ LU, RK, BC ] = update_matrices_solute( LU, RK, BC, BC_prev, BC_cur, R,gama,RA, RD, LA, LD, fi, dt)
  %UPDATE_MATRICES function updates the matrices each time step according to the sources and sinks
  % and boundary conditions
  LR = dt * gama .* fi(2:end) ;
  RR = dt * (1 - gama) .* fi(2:end);

  LU(eye(size(LU))~=0) = diag(LU) - LR .* R;
  RK(eye(size(RK))~=0) = diag(RK) + RR .* R;
  BC(1) = (RA(1)+RD(1)) * BC_prev + (LA(1) + LD(1)) * BC_cur;
end

function [ C ] = solving_eq( LU, RK,R_eq_non_lin, BC,C0, fi)
  %SOLVING_EQ S
  
  C = LU \ ( RK * C0 + R_eq_non_lin + BC );
  C = (C >= 0) .* C; % Just to be on the safe side. This is a little hack that concentration can not be negative
end