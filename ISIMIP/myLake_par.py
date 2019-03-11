import os

def mylakepar(longitude, latitude, outpath,swa_b1=0.1,k_BOD=0.01,k_SOD=100,I_scDOC=1):
    """
    Creates MyLake parameter file. If the file LAE_para_all1.txt is present, it will be used to prepare the parameters.
    Otherwise, the string in this function while be used.

    :param longitude: Type int. Longitude coordinate of Mylake in degrees.
    :param latitude: Type int. Latitude coordinate of Mylake in degrees
    :param outpath: Type str. Filename where a file of Mylake parameters will be written. In a typical run, this is the
    return value of mylakeinit function
    :return: None
    """

    if (os.path.isfile ( "LAE_para_all1.txt" )): #this file allows change of the four coefficients, if nothing is given, will uses initial values
        print('using file')
        with open ( "LAE_para_all1.txt", "r" ) as infile:
            out = infile.read () % (latitude, longitude, I_scDOC, swa_b1, k_BOD, k_SOD)

    else:
        out = '''-999	"Mylake parameters"			
    Parameter	Value	Min	Max	Unit
    dz	1.0	0.5	2	m
    Kz_ak	0.007	NaN	NaN	(-)
    Kz_ak_ice	0.003	NaN	NaN	(-)
    Kz_N0	7.00E-05	NaN	NaN	s-2
    C_shelter	NaN	NaN	NaN	(-)
    latitude	%.5f	NaN	NaN	dec.deg
    longitude	%.5f	NaN	NaN	dec.deg
    alb_melt_ice	0.6	NaN	NaN	(-)
    alb_melt_snow	0.9	NaN	NaN	(-)
    PAR_sat	3.00E-05	1.00E-05	1.00E-04	mol m-2 s-1
    f_par	0.89	NaN	NaN	(-)
    beta_chl	0.015	0.005	0.045	m2 mg-1
    lamgbda_I	5	NaN	NaN	m-1
    lambda_s	15	NaN	NaN	m-1
    sed_sld	0.36	NaN	NaN	(m3/m3)
    I_scV 	1.339	NaN	NaN	(-)
    I_scT	1.781	NaN	NaN	deg C
    I_scC	1	NaN	NaN	(-)
    I_scS	1	1.1	1.9	(-)
    I_scTP	1	0.4	0.8	(-)
    I_scDOP	1	NaN	NaN	(-)
    I_scChl	1	NaN	NaN	(-)
    I_scDOC	%s	NaN	NaN	(-)
    swa_b0	0.727	NaN	NaN	m-1
    swa_b1	%s	0.8	1.3	m-1
    S_res_epi	3.30E-07	7.30E-08	1.82E-06	m d-1 (dry mass)
    S_res_hypo	3.30E-08	NaN	NaN	m d-1 (dry mass)
    H_sed	0.03	NaN	NaN	m
    Psat_Lang	2500	NaN	NaN	mg m-3
    Fmax_Lang	8000	5000	10000	mg kg-1
    Uz_Sz	0.3	0.1	1	m d-1
    Uz_Chl	0.16	0.05	0.5	m d-1
    Y_cp	1	NaN	NaN	(-)
    m_twty	0.2	0.1	0.3	d-1
    g_twty	1.5	1	1.5	d-1
    k_sed_twty	2.00E-04	NaN	NaN	d-1
    k_dop_twty	0	NaN	NaN	d-1
    P_half	0.2	0.2	2	mg m-3
    PAR_sat2	3.00E-05	NaN	NaN	mol m-2 s-1
    beta_chl2	0.015	NaN	NaN	m2 mg-1
    Uz_Chl2	0.16	NaN	NaN	m d-1
    m_twty2	0.2	NaN	NaN	d-1
    g_twty2	1.5	NaN	NaN	d-1
    P_half2	0.2	NaN	NaN	mg m-3
    oc_DOC	0.01	NaN	NaN	m2 mg-1
    qy_DOC	0.1	NaN	NaN	mg mol-1
    k_BOD	%s	NaN	NaN	d-1
    k_SOD	%s	NaN	NaN	mg m-2
    theta_BOD	1.047	NaN	NaN	(-)
    theta_BOD_ice	1.13	NaN	NaN	(-)
    theta_SOD	1	NaN	NaN	(-)
    theta_SOD_ice	1	NaN	NaN	(-)
    theta_T	4	NaN	NaN	deg.celcius
    pH	5.2	NaN	NaN	(-)
    I_scDIC	1	NaN	NaN	(-)
    Mass_Ratio_C_Chl	100	NaN	NaN	(-)
    SS_C	0.25	NaN NaN 57
    density_org_H_nc	1.95	NaN NaN 58
    density_inorg_H_nc	2.65	NaN NaN 59
    I_scO	1	NaN NaN (-)
    ''' % (latitude, longitude, I_scDOC, swa_b1, k_BOD, k_SOD)


    with open(outpath, 'w') as f:
        f.write(out)
