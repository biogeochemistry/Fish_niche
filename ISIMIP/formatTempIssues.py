
import pandas as pd
import os
# taupo, klicava, waahi
lakelist = ['Rappbode',"Alqueva", "Argyle", 'Bourget', 'CrystalBog', 'Eagle', 'Ekoln', 'EsthwaiteWater', 'FallingCreek',
'Feeagh', 'GreatPond', 'Klicava', 'Kuivajarvi','Monona', 'MtBold', 'Muggelsee', 'Ngoring',
'NohipaloMustjarv', 'NohipaloValgejarv',  'Rimov', 'Sammamish', 'Sau',  'Taupo',
'Vendyurskoe', 'Vortsjarv', 'Waahi', 'Washington', "Windermere"]

regions = {"US": ["Allequash", "Annie", "BigMuskellunge", "BlackOak", "Crystal", "CrystalBog", "Delavan", "FallingCreek", "Fish", "GreatPond", "Laramie", "Mendota", "Monona",
                  "Okauchee", "Sammamish", "Sparkling", "Sunapee", "Tahoe", "Toolik", "Trout", "TroutBog", "TwoSisters",
                  "Washington", "Wingra"],
           "CH": ["Biel", "LowerZurich", "Neuchatel"],
           "PT": ["Alqueva"],
           "FR": ["Annecy", "Bourget", "Geneva"],
           "AU": ["Argyle", "BurleyGriffin", "MtBold"],
           "CA": ["Dickie", "Eagle", "Harp"],
           "SE": ["Ekoln", "Erken"],
           "UK": ["EsthwaiteWater", "Windermere"],
           "IE": ["Feeagh"],
           "FI": ["Kilpisjarvi", "Kuivajarvi", "Paajarvi"],
           "IL": ["Kinneret"],
           "RW": ["Kivu"],
           "CZ": ["Klicava", "Rimov", "Zlutice"],
           "NO": ["Langtjern"],
           "RU": ["Mozhaysk", "Vendyurskoe"],
           "DE": ["Muggelsee", "Rappbode", "Stechlin"],
           "CN": ["Ngoring"],
           "EE": ["NohipaloMustjarv", "NohipaloValgejarv", "Vortsjarv"],
           "ES": ["Sau"],
           "NZ": ["Rotura", "Tarawera", "Taupo", "Waahi"]}
'''
Get a list of keys from dictionary which has the given value
'''


def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for items in listOfItems:
        if len(listOfKeys) != 0:
            break
        else:
            for item in items[1]:
                if item == valueToFind:
                    listOfKeys.append(items[0])
                    break
    return listOfKeys


if __name__ == '__main__':
    for lake in lakelist:
        file = "%s/temp.csv" % (lake)
        if os.path.exists(file) is True:
            print(lake)
            data = pd.read_csv(file)
            data['depth'] = data['depth'].astype(float)
            data = data.sort_values(by=['datetime', 'depth'])
            key = getKeysByValue(regions, lake)[0]
            print(key)
            setdata = {"SITE_ID": list(data['depth']), "SITE_NAME": list(data['depth']), "TIMESTAMP": list(data['depth']), "DEPTH": list(data['depth']), "WTEMP": list(data['temp'])}
            dataframe = pd.DataFrame(setdata)

            dataframe['SITE_ID'] = '%s-%s' % (key, lake)
            dataframe['SITE_NAME'] = lake
            dataframe['TIMESTAMP'] = pd.to_datetime(data['datetime']).apply(lambda x: x.strftime('%Y%m%d'))
            dataframe.to_csv(r'C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\observations\%s\%s\%s_temp_daily.csv' % (key, lake, lake), index=False)

