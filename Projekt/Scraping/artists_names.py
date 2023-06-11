import billboard

# Function that returns a list of top artists from a given year with their rank
def get_top_artists_from_year(year : int) -> list[tuple[int, str]]:
    chart = billboard.ChartData('top-artists', year=year)
    top_artists = list()
    for artist in chart:
        top_artists.append((artist.rank, artist.artist))
    return top_artists

# Function that returns a list of top artists from a given year range with their rank
def get_top_artists_from_year_range(start_year : int, end_year : int) -> list[tuple[int, str]]:
    top_artists = list()
    for year in range(start_year, end_year):
        top_artists.append(get_top_artists_from_year(year))
    return [item for sublist in top_artists for item in sublist]

# Function that returns a list of names of the artists without any duplicates in alphabetical order
def filter_artists(start_year : int, end_year : int) -> list[str]:
    top_artists = get_top_artists_from_year_range(start_year, end_year)
    names = [tup[1] for tup in top_artists]
    return sorted(list(set(names)))
    
    