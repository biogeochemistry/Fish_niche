import gzip
import shutil
import os.path
#2018-07-30 MC 4 variables to change. in order :
            #   path_to_zip_file:           directory where .gz files are.
            #   directiory_to_extract_to:   directory where .h5 will be extract
            #   listeofscenarios:           list of selected scenarios. see scenarios below.
            #   listofmodels:               list of selected models. see models below.
            #   variables:                  list of selected variables

path_to_zip_file = "G:\cordex"
directory_to_extract_to = r"G:\cordex"
listofscenarios = [2]
listofmodels = [4]
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


url1 = 'http://ns2806k.web.sigma2.no/'  # 5-9-2018 MC
url2 = '_EUR-11_'

def delete_h5(v, model, scenario,cordexfolder,reportfile):
    m0, m1 = models[model]
    s0, s1 = scenarios[scenario]
    urlsA = ['%s/Lakes_%s%s%s_%s_%s_%s-%s.h5' %  # MC 2017-05-16 add of "Lakes_" to be fix h5 filename on repertory, erase _day
        (cordexfolder, v, url2, m0, s0[0], m1, s0[1], s0[2])]

    urlsB = ['%s/Lakes_%s%s%s_%s_%s_%s-%s.h5' %  # MC 2017-05-16 add of "Lakes_" to be fix h5 filename on repertory
             ( cordexfolder,v, url2, m0, s1[0], m1, s1[1], s1[2])]

    urls = urlsA + urlsB
    new_urls = []
    for url in urls:
        if url.find ( "MOHC-HadGEM2-ES" ) != -1:
            head, _sep, tail = url.rpartition ( '1' )
            url = head + "0" + tail
        new_urls.append ( url )
    urls = new_urls

    for file in urls:
        print ( 'trying to delete %s' % file )
        with open ( reportfile, 'a' ) as f:
            f.write ( 'trying to delete %s\n' % file )
            f.close ()
        if os.path.isfile(file):
            try:
                os.remove ( file )
                with open ( reportfile, 'a' ) as f:
                    f.write ( 'file deteted\n' )
                    f.close ()
            except OSError as e:  # name the Exception `e`
                with open ( reportfile, 'a' ) as f:
                    f.write ( '\nFailed with: %s\t' % e.strerror )
                    f.write('Error code: %s\n'% e.code)
                    f.close ()
                print("Failed with:", e.strerror ) # look what it says
                print("Error code:", e.code)
        else:
            with open ( reportfile, 'a' ) as f:
                f.write ( 'file not existed in %s \n'% cordexfolder )
                f.close ()


def unzip(v, model, scenario,reportfile):
    m0,m1 = models[model]
    s0,s1 = scenarios[scenario]
    urlsA = ['/Lakes_%s%s%s_%s_%s_%s-%s.h5' % #MC 2017-05-16 add of "Lakes_" to be fix h5 filename on repertory, erase _day
             (v, url2, m0, s0[0], m1, s0[1], s0[2])]
             #for v in variables]

    urlsB = ['/Lakes_%s%s%s_%s_%s_%s-%s.h5' % #MC 2017-05-16 add of "Lakes_" to be fix h5 filename on repertory
             ( v, url2, m0, s1[0], m1, s1[1], s1[2])]
             #for v in variables]
    urls =  urlsA +urlsB

    ##for models 4, dates end by 30 and not 31 MC 2018-05-18
    new_urls = []
    for url in urls:
        if url.find("MOHC-HadGEM2-ES") != -1:
            head,_sep,tail = url.rpartition('1')
            url = head+"0"+tail
        new_urls.append(url)
    urls = new_urls
    i = 1

    for file in urls:
        #file = os.path.join ( cordexfolder, file )
        file_in = "%s%s.gz"%(path_to_zip_file,file)
        file_out =  "%s%s"%(directory_to_extract_to,file)
        print('trying to unzip %s'%file_in)
        with open(reportfile,'a') as f:
            f.write('trying to unzip %s\n'%file_in)
            f.close()
        if os.path.isfile(file_in):
            if not os.path.isfile ( file_out ):
                with gzip.open ( file_in, 'rb' ) as f_in:
                    with open ( file_out, 'wb' ) as f_out:
                        shutil.copyfileobj ( f_in, f_out )
                        if os.path.isfile(file_out):
                            print ( '%s created' % file_in )
                            with open ( reportfile, 'a' ) as f:
                                f.write ( '%s created\n' % file_in )
                                f.close ()
                        else:
                            print ( 'ERROR, %s is not created' % file_in )
                            with open ( reportfile, 'a' ) as f:
                                f.write ( 'ERROR, %s is not created\n' % file_in )
                                f.close ()
            else:
                print('already exist')
        else:
            print ( "%s doesn't exist" % file_in )
            with open ( reportfile, 'a' ) as f:
                f.write ( "%s doesn't exist\n" % file_in )
                f.close ()



if __name__ == '__main__':

    for scenario in listofscenarios:
        for model in listofmodels:
            for v in variables:
                unzip(v,model,scenario,"rapport_unzip.txt")
    #num_cores = multiprocessing.cpu_count ()
    #model = 4
    #scenario = 2
    #Parallel ( n_jobs=num_cores ) ( delayed ( unzip ) ( v, model, scenario,cordexfolder="1" ) for v in variables )