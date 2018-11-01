"""
unzip_h5_files_function.py
last modification: 2018-08-30 MC

Unzip all files need for the modelization depending the variables, the models and the scenarios used
Write report "rapport_unzip.txt" which detail the run.
Possible to use command line by:  python unzip_h5_files_function.py <[list of models]> <[list of scenarios]> <"directory of .gz files"> <"directory where go final .h5 files">
or change variable on line 139 and below

return .h5 files and report file in the directory where go final .h5 files
"""

import os
import os.path
import sys
#from joblib import Parallel, delayed #2018-07-30 MC uncomment if parallel is used
#import multiprocessing #2018-07-30 MC uncomment if parallel is used
import gzip
import shutil

variables = ['clt', 'hurs', 'tas', 'rsds', 'ps', 'pr', 'sfcWind']

scenarios = {1: (('historical', '19710101', '19751231'), ('historical', '19760101', '19801231')),
             2: (('historical', '20010101', '20051231'), ('rcp45', '20060101', '20101231')),
             3: (('rcp45', '20310101', '20351231'), ('rcp45', '20360101', '20401231')),
             4: (('rcp45', '20610101', '20651231'), ('rcp45', '20660101', '20701231')),
             5: (('rcp45', '20910101', '20951231'), ('rcp45', '20960101', '21001231')),
             6: (('rcp85', '20310101', '20351231'), ('rcp85', '20360101', '20401231')),
             7: (('rcp85', '20610101', '20651231'), ('rcp85', '20660101', '20701231')),
             8: (('rcp85', '20910101', '20951231'), ('rcp85', '20960101', '21001231'))}

models = {1: ('ICHEC-EC-EARTH', 'r1i1p1_KNMI-RACMO22E_v1_day'),  # MC 2018-05-16
          2: ('ICHEC-EC-EARTH', 'r3i1p1_DMI-HIRHAM5_v1_day'),
          3: ('MPI-M-MPI-ESM-LR', 'r1i1p1_CLMcom-CCLM4-8-17_v1_day'),
          4: ('MOHC-HadGEM2-ES', 'r1i1p1_SMHI-RCA4_v1_day'),
          5: ('IPSL-IPSL-CM5A-MR', 'r1i1p1_IPSL-INERIS-WRF331F_v1_day'),
          6: ('CNRM-CERFACS-CNRM-CM5', 'r1i1p1_CLMcom-CCLM4-8-17_v1_day')}

url1 = 'http://ns2806k.web.sigma2.no/'  # 5-9-2018 MC new site containing all .zip files
url2 = '_EUR-11_'

def loop_throught_unzip_files(variables, listofmodels, listofscenarios, path_to_zip_file, directory_to_extract_to):
    """
    function that unzip all files need for the modelization depending the variables, the models and the scenarios used
    :param listofmodels: list of model used
    :param listofscenarios: list of scenario used
    :param variables: list of variable used
    :param path_to_zip_file: directory where .gz files are
    :param directory_to_extract_to: directory where .h5 will be extract

    :return: .h5 files and report of the run

    .. note:: The .h5 files need to be first download if the directory of .gz files don't exist. Site: 'http://ns2806k.web.sigma2.no/'
    """

    reportfile = "rapport_unzip.txt"
    for scenario in listofscenarios:
        for model in listofmodels:
            for v in variables:
                m0, m1 = models[model]
                s0, s1 = scenarios[scenario]
                urlsA = [
                    '/Lakes_%s%s%s_%s_%s_%s-%s.h5' %  # MC 2017-05-16 add of "Lakes_" to be fix h5 filename on repertory, erase _day
                    (v, url2, m0, s0[0], m1, s0[1], s0[2])]


                urlsB = ['/Lakes_%s%s%s_%s_%s_%s-%s.h5' %  # MC 2017-05-16 add of "Lakes_" to be fix h5 filename on repertory
                    (v, url2, m0, s1[0], m1, s1[1], s1[2])]

                urls = urlsA + urlsB

                ##for models 4, dates end by 30 and not 31 MC 2018-05-18
                new_urls = []
                for url in urls:
                    if url.find ( "MOHC-HadGEM2-ES" ) != -1:
                        if  url.find("20960101")!= -1: #2018-07-31 MC for model 4, scenario 5, interval end = 2099-11-30, not 2100-12-31 and scenario 8 = 2099-12-30
                            if url.find("rcp45") != -1:
                                head,_sep,tail = url.rpartition('21001231')
                                url= head + "20991130" + tail
                            elif url.find("rcp85") != -1:
                                head, _sep, tail = url.rpartition ( '21001231' )
                                url = head + "20991230" + tail
                        else:
                            head, _sep, tail = url.rpartition ( '1' )
                            url = head + "0" + tail
                    new_urls.append ( url )
                urls = new_urls

                for file in urls:
                    #file = os.path.join ( cordexfolder, file ) # 2018-07-30 MC replace gzip application (called with os.system) by gzip function(integrated to python)
                    #print('trying to unzip %s'%file)
                    #with open(reportfile,'a') as f:
                    #    f.write('trying to unzip %s\n'%file)
                    #    f.close()
                    #command = ['gzip -d -k %s'%file]
                    #os.system ( ' & '.join(command) )
                    file_in = "%s%s.gz" % (path_to_zip_file, file)
                    file_out = "%s%s" % (directory_to_extract_to, file)
                    #print ( 'trying to unzip %s' % file_in )
                    with open ( reportfile, 'w' ) as f:
                        f.write ( 'trying to unzip %s\n' % file_in )
                        f.close ()
                    if os.path.isfile ( file_out ):
                        print ( '%s already exist' % file_out )
                        with open ( reportfile, 'a' ) as f:
                            f.write ( '%s already exist\n' % file_out )
                            f.close ()
                    else:
                        if os.path.isfile ( file_in ):
                            with gzip.open ( file_in, 'rb' ) as f_in:
                                with open ( file_out, 'wb' ) as f_out:
                                    shutil.copyfileobj ( f_in, f_out )
                                if os.path.isfile ( file_out ):
                                    print ( '%s created' % file_out )
                                    with open ( reportfile, 'a' ) as f:
                                        f.write ( '%s created\n' % file_out )
                                        f.close ()
                                else:
                                    print ( 'ERROR, %s is not created' % file_in )
                                    with open ( reportfile, 'a' ) as f:
                                        f.write ( 'ERROR, %s is not created\n' % file_in )
                                        f.close ()
                        else:
                            print ( "%s doesn't exist" % file_in )
                            with open ( reportfile, 'a' ) as f:
                                f.write ( "%s doesn't exist\n" % file_in )
                                f.close ()

if __name__ == '__main__':
    # 2018-07-30 MC 5 variables that need to be change. in order :
    #   path_to_zip_file:           directory where .gz files are.
    #   directiory_to_extract_to:   directory where .h5 will be extract
    #   listeofscenarios:           list of selected scenarios. see scenarios below.
    #   listofmodels:               list of selected models. see models below.
    #   variables:                  list of selected variables

    #path_to_zip_file = "G:\lakes_gz"
    #directory_to_extract_to = "E:\optimisation\cordex"
    #listofscenarios = [1,2,3,4,5,6,7,8]
    #listofmodels = [1,2,3,4,5,6]
    #variables = ['clt', 'hurs', 'tas', 'rsds', 'ps', 'pr', 'sfcWind']

    listofmodels = [int ( s ) for s in sys.argv[1].split ( ',' )]
    listofscenarios = [int ( s ) for s in sys.argv[2].split ( ',' )]
    path_to_zip_file = sys.argv[3]
    directory_to_extract_to = sys.argv[4]

    loop_throught_unzip_files ( variables, listofmodels, listofscenarios, path_to_zip_file, directory_to_extract_to )

