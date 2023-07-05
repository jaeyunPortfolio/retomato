import Part_1 
import re
import requests
from bs4 import BeautifulSoup

import time

movielist = {'star_wars_the_last_jedi', 'captain_marvel', 'terminator_dark_fate', 'suffragette','the_danish_girl_2015'}

delay=1

def crawl_by_list(movielist):

    for movie in movielist:
        Part_1.make_reviews(movie)
        print(movie, 'Done')
        time.sleep(delay)



def crawl_by_list_search(movielist):



    for movie in movielist:

        url = 'http://www.rottentomatoes.com/search?search=' + movie
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')

        txttmp = soup.find('search-page-media-row')
        if txttmp!=None:
            tmp = str(txttmp.select_one('a')).split()[3]
            title = tmp[39:-1]
            if title!= None:
                Part_1.make_reviews(title)
                time.sleep(delay)
                print(title, 'Done')



def crawl_sp():

    url = 'https://editorial.rottentomatoes.com/guide/essential-movies-to-watch-now/'
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')

    txttmp = soup.find_all('div', {'class': 'row countdown-item'})

    for ttmp in txttmp:
        tmp = str(ttmp.select_one('a')).split()[2]
        title = tmp[39:-2]
        if title!= None:
            Part_1.make_reviews(title)
            time.sleep(delay)
            print(title, 'Done')



    url='https://editorial.rottentomatoes.com/guide/essential-movies-to-watch-now/2/'
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')

    txttmp = soup.find_all('div', {'class': 'row countdown-item'})

    for ttmp in txttmp:
        tmp = str(ttmp.select_one('a')).split()[2]
        title = tmp[39:-2]
        if title!= None:
            Part_1.make_reviews(title)
            time.sleep(delay)
            print(title, 'Done')

#Part_1.clean_both()
#crawl_by_list(movielist)
#crawl_sp()
Part_1.to_data_csv()