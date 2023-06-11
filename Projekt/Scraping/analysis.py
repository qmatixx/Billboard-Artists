import pandas as pd
from pandas_profiling import ProfileReport

if __name__ == "__main__":
    plik = 'artists.csv'
    data = pd.read_csv(plik, sep=',', index_col=[0])
    print(data.head(20))  
    profile = ProfileReport(data, title='Arists Raport')
    profile.to_file("artists_raport.html")