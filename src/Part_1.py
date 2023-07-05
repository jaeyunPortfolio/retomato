#https://github.com/sjmiller8182/tomatopy  참조함


import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, List
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import csv



import os 
from sklearn.preprocessing import OneHotEncoder

#NLP tools
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient



HOST = 'cluster0.5jop1a2.mongodb.net'
USER = 'kimzakga1'
PASSWORD = 'codestates'
DATABASE_NAME = 'proj4'
MONGO_URI = f"mongodb+srv://{USER}:{PASSWORD}@{HOST}/{DATABASE_NAME}?retryWrites=true&w=majority"


#file_path=os.getcwd()+'/src/'
file_path='src/'
#file_path=''

# run once on import

"""
Part 1

영화 리뷰 스크레이핑을 위한 함수들을 구현해봅니다.

사용할 사이트는 네이버의 영화 리뷰입니다.

> 해당 BASE_URL 은 아래 설정이 되어 있습니다. 코드를 작성하면서 `BASE_URL` 에 문자열을 추가로 붙여서 사용해보세요!
> 추가로 BASE_URL 은 접속이 되지 않습니다.
"""

BASE_URL = "https://www.rottentomatoes.com/"


def get_page(page_url):
    """
    get_page 함수는 페이지 URL 을 받아 해당 페이지를 가져오고 파싱한 두
    결과들을 리턴합니다.

    예를 들어 `page_url` 이 `https://github.com` 으로 주어진다면
        1. 해당 페이지를 requests 라이브러리를 통해서 가져오고 해당 response 객체를 page 변수로 저장
        2. 1번의 response 의 html 을 BeautifulSoup 으로 파싱한 soup 객체를 soup 변수로 저장
        3. 저장한 soup, page 들을 리턴하고 함수 종료

    파라미터:
        - page_url: 받아올 페이지 url 정보입니다.

    리턴:
        - soup: BeautifulSoup 으로 파싱한 객체
        - page: requests 을 통해 받은 페이지 (requests 에서 사용하는 response
        객체입니다).
    """


    page = requests.get(page_url)
    soup = BeautifulSoup(page.text, 'html.parser')


    return soup


def make_score(reviews, movie_title):
    """
    
    ------------------------------------------------
    """
    avg = 0
    hs=0
    of=0
    pure=0
    hssum=0
    ofsum=0
    puresum=0

    df = pd.DataFrame(reviews)
    df['ofscore'] = Ofdetector(df['review_text'])
    df['hsscore'] = Hsdetector(df['review_text'])


    for i in range(len(reviews)):  
        avg+=df['review_star'][i]
        hs+=df['review_star'][i]*df['hsscore'][i]
        of+=df['review_star'][i]*df['ofscore'][i]
        pure+=df['review_star'][i]*(1-df['ofscore'][i])*(1-df['ofscore'][i])

        hssum+=df['hsscore'][i]
        ofsum+=df['ofscore'][i]
        puresum+=(1-df['ofscore'][i])*(1-df['ofscore'][i])


    avg /= len(reviews)
    if hssum!=0: hs /= hssum
    if ofsum!=0: of /= ofsum
    if puresum!=0: pure /= puresum

    score=dict()
    score['title']= movie_title
    score['Average']= avg
    score['HateSpeech']= hs
    score['Offensive']= of
    score['Pure']= pure

    score['HateSpeechnum']= hssum
    score['Offensivenum']= ofsum
    score['Purenum']= puresum

    return score


def _build_url(m_name: str, m_type: str = 'Movie', sep: str = '_') -> str:
    """Builds url for main page of movie
    Parameters
    ----------
    m_name : str
        The url to scrape from RT
    m_type : str
        Only "Movie" is supported now
    sep : str
        Word seperator to use '-' or '_' typically
    Returns
    -------
    bs4 object
        hrml content from bs4 html parser
    """
    
    # TODO: add tv show selection
    if m_type == 'Movie':
        url = BASE_URL + 'm/' + _format_name(m_name, sep) + '/'
    else:
        raise Exception('Argument `m_type` must be `Movie`')
    return url

def _format_name(m_name: str, sep: str = '_') -> str:
    """Formats name for url
    Parameters
    ----------
    m_name : str
        Name of movie
    sep : str
        Word seperator to use '-' or '_' typically
    Returns
    -------
    str
        movie name formatted for url insertion
    """
    
    # enforce lower case
    m_name = m_name.lower()
    
    # remove any punctuation
    remove_items = "'-:,"
    for i in remove_items:
        if i in m_name:
            m_name = m_name.replace(i,'')
    m_name = m_name.strip('"')
    return m_name.replace(' ', sep)


def make_reviews(movie_title, insert_review='Fasle'):
    """
    
    """
    movie_url = _build_url(movie_title)
    soup= get_page(movie_url)

    if soup!='':
        txttmp = soup.find('score-board')
        if txttmp !=None:
            info=dict()
            info=get_main_page_info(soup, movie_title)
            reviews=list()
            reviews=get_user_reviews(movie_url)
            score = dict()
            score = make_score(reviews, movie_title)
            info2=dict()
            info2=score
            info2.update(info)

            insert_info(info2)
            if insert_review=='True':
                insert_reviews(reviews, movie_title)
            insert_score(score)
        else:
            print("soup nonexists")

    else:
        print("soup nonexists")
    
    '''
    review_list = []

    raw_reviews = soup.find(
        'table', {'class': 'list_netizen'}).findChildren('tr')[1:]

    for rr in raw_reviews:
        raw_review_text = rr.select('td > a')[1].attrs['onclick']
        parsed_text = eval(re.search(r'[(].*[)]', raw_review_text)[0])[2]
 
        raw_review_score = str(rr.select('em'))
        parsed_score = int(re.sub(r'[^0-9]', '', raw_review_score))
       
        result = {}
        result['review_text'] = parsed_text
        result['review_star'] = parsed_score
        review_list.append(result)

    return review_list
    '''


#===============
# user functions
#===============

def get_main_page_info(soup, movie_title) -> Dict[str, List]:
    """Scrapes info from a movie main page
    Parameters
    ----------
    page : str
        The url to scrape from RT
    Returns
    -------
    dict
        dict of scraped info with keys:
        synopsis, rating, genre, studio, director, writer, currency,
        box_office, runtime
    """
    
    info = dict()   
    
        
    if soup == '':
        # return None when failed soup; None is easy to detect
        return None
        
    else:
        # prepare soup
        # ignore steping through tree due to instability
        info_html = str(soup)
        
        ### eat soup ###
        
        # get synopsis
        info['title']=movie_title

        txttmp = soup.find('h1', {'data-qa' : "score-panel-movie-title" })
        
        if txttmp != None:
            txttmp=re.sub('  ', '',txttmp.text)
            txttmp=re.sub('\n', '',txttmp)       
            info['full_title'] = txttmp
        else:
            info['full_title'] = None


        txttmp = soup.find('p', {'data-qa' : "movie-info-synopsis" })
        if txttmp != None:
            txttmp=re.sub('  ', '',txttmp.text)
            txttmp=re.sub('\n', '',txttmp)       
            info['synopsis'] = txttmp
        else:
            info['synopsis'] = None
        

          
        # get rating
        txttmp = soup.find('score-board')
        if txttmp != None:
            txttmp = txttmp.attrs['tomatometerscore']
            info['tomatometerscore'] = txttmp
        else:
            info['tomatometerscore'] = None

        txttmp = soup.find('score-board')
        if txttmp != None:
            txttmp=txttmp.attrs['audiencescore']
            info['audiencescore'] = txttmp
        else:
            info['audiencesocre'] = None
      
        '''
        txttmp = soup.find('script', {'id' : "score-details-json" })
        if len(txttmp) > 0:
          txttmp = txttmp.text
          loca= txttmp.find('averageRating')
          txttmp=int(re.sub(r'[^0-9]', '', txttmp[loca+15:loca+25]))
          txttmp/=10
          info['AverageRating'] = txttmp
        else:
          info['AverageRating'] = None
        '''

        # get genre
        labeltmp = soup.find_all('li', {'data-qa' : "movie-info-item" })
        infotmp=dict()
        for x in labeltmp:
            
            label = x.select_one('p > b').text
            cont = x.select_one('p > span')
            infotmp[label]=cont


        txttmp=infotmp.get("Genre:")
        if txttmp != None:
            genre = list()
            temp = re.sub('\n', '',txttmp.text)
            temp = re.sub(' ', '',temp)
            genre = re.sub(',', ' ',temp).split()
            info['genre'] = '|'.join(genre)
        else:
            info['genre'] = None
        
        # get director
        txttmp=infotmp.get("Director:")
        if txttmp != None:
            director = list()
            temp = re.sub('\n', '',txttmp.text)
            temp = re.sub(' ', '',temp)
            director = re.sub(',', ' ',temp).split()
            
            info['director'] = '|'.join(director)
        else:
            info['director'] = None
        
        # get writer
        txttmp=infotmp.get("Writer:")
        if txttmp != None:
            writer = list()
            temp = re.sub('\n', '',txttmp.text)
            temp = re.sub(' ', '',temp)
            writer = re.sub(',', ' ',temp).split()
            info['writer'] = '|'.join(writer)
        else:
            info['writer'] = None
        
        # get dates
        txttmp=infotmp.get("Release Date (Theaters):")
        if txttmp != None:
            temp = txttmp.select_one('time').text
            temp = re.sub('\n', '',temp)
            temp = re.sub(' ', '',temp)
            info['theater_date'] = temp           
        else:
            info['theater_date'] = None
        
        txttmp=infotmp.get("Release Date (Streaming):")
        if txttmp != None:
            temp = txttmp.select_one('time').text
            temp = re.sub('\n', '',temp)
            temp = re.sub(' ', '',temp)
            info['dvd/streaming_date'] = temp           
        else:
            info['dvd/streaming_date'] = None
        
        # get box_office
        txttmp=infotmp.get("Box Office (Gross USA):")
        if txttmp != None:
            temp = re.sub('\n', '',txttmp.text)
            temp = re.sub(' ', '',temp)
            info['box_office'] = temp
        else:
            info['box_office'] = None
        
        # get runtime
        txttmp=infotmp.get("Runtime:")
        if txttmp != None:
            temp = re.sub('\n', '',txttmp.text)
            temp = re.sub(' ', '',temp)
            info['runtime'] = temp
        else:
            info['runtime'] = None
        

        # get studio
        txttmp=infotmp.get("Distributor:")
        if txttmp != None:
            studio = list()
            temp = re.sub('\n', '',txttmp.text)
            temp = re.sub(' ', '',temp)
            studio = re.sub(',', ' ',temp).split()
            info['Distributor'] = '|'.join(studio)
        else:
            info['Distributor'] = None
        

            # get franchise
        txttmp=infotmp.get("View the collection:")
        if txttmp != None:
            addi = list()
            temp = re.sub('\n', '',txttmp.text)
            temp = re.sub(' ', '',temp)
            addi = re.sub(',', ' ',temp).split()
            info['Collection'] = '|'.join(addi)
        else:
            info['Collection'] = None
            
        return info



def _get_user_reviews_from_page(soup):
    """Get the review, rating, critic, if critic is a 
    'top critic', publisher, date from a given page (bs4)
    Parameters
    ----------
    soup : bs4 object
        bs4 html tree from html_parser
    Returns
    -------
    list
        list of lists containing the following:
        reviews, rating, fresh, critic, top_critic,
        publisher, date
        
    """


    review_list = []




    raw_reviews = soup.find_all('div', {'class': "review-text-container"})




    for rr in raw_reviews:
        raw_review_text = rr.select_one('drawer-more > p').text
        parsed_text = re.sub('\n', '',raw_review_text)
        parsed_text = re.sub('  ', '',parsed_text)

        raw_score = str(rr.select_one('span'))
        parsed_score = 5-raw_score.count("star-display__half")/2-raw_score.count("star-display__empty")

       
        result = {}
        result['review_text'] = parsed_text
        result['review_star'] = parsed_score
        review_list.append(result)

    return review_list

#===============
# user functions
#===============
    
def get_user_reviews(page: str) -> Dict[str, List]:
    """Crawls the set of critic review pages for the given movie.
    Returns a dict withkeys: reviews, rating, fresh,
    critic, top_critic, publisher, date.
    Parameters
    ----------
    page : str
        main page url for movie
    Returns
    -------
    dict
        dict containing scraped review info with the following keys:
        'reviews', 'rating', 'fresh', 'critic', 'top_critic',
        'publisher', 'date'
        
    """

    # containers
    reviews = []
        
    # make soup
    soup = get_page(page + "/reviews?type=user")
    
    # how many soups?
    
    if soup is not None:

        #nxt = soup.find('pagination-manager').attrs['hasnext']

        # eat soup
        for page_num in range(0, 1):
            soup = get_page(page + "/reviews?type=user&sort=&page=" + str(page_num))#######            
            reviews.extend(_get_user_reviews_from_page(soup))

        
    else:
        # if pages doesnt match return None; its easy to detect
        reviews= None
        
    return reviews
    



def Hsdetector(input):
    new = LogisticRegression(random_state = 0)

    

    cv = CountVectorizer(max_features = 2000)
    with open(file_path+'countvectorizer.pkl', 'rb') as f:
        cv = pickle.load(f)

    corpus2=[]
    for i in range(0, len(input)):
        review = re.sub('[^a-zA-Z]', ' ', input[i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus2.append(review)

    X = cv.transform(corpus2).toarray()
    



    with open(file_path+"lrhatespeech_hate.pkl", 'rb') as file:
        new = pickle.load(file)
    #Linear Regression
    result = new.predict(X)
    return result

def Ofdetector(input):
    new = LogisticRegression(random_state = 0)



    cv = CountVectorizer(max_features = 2000)
    with open(file_path+"countvectorizer.pkl", 'rb') as f:
        cv = pickle.load(f)

    corpus2=[]
    for i in range(0, len(input)):
        review = re.sub('[^a-zA-Z]', ' ', input[i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus2.append(review)

    X = cv.transform(corpus2).toarray()
    


    

    with open(file_path+"lrhatespeech_offensive.pkl", 'rb') as file:
        new = pickle.load(file)
    #Linear Regression
    result = new.predict(X)
    return result





def insert_info(info):
    COLLECTION_NAME = 'info'

    client = MongoClient({MONGO_URI})
    database = client[DATABASE_NAME]
    Collection = database[COLLECTION_NAME]

    movie_title=info['title']
    Collection.delete_many({'title':movie_title})
    Collection.insert_one(info)
    
    client.close()

def get_info(movie_title):
    COLLECTION_NAME = 'info'

    client = MongoClient({MONGO_URI})
    database = client[DATABASE_NAME]
    Collection = database[COLLECTION_NAME]

    info = Collection.find_one({'title':movie_title}, {'_id':0})
    
    client.close()
    return info


def insert_reviews(reviews, movie_title):
    COLLECTION_NAME = movie_title

    client = MongoClient({MONGO_URI})
    database = client[DATABASE_NAME]
    Collection = database[COLLECTION_NAME]
    Collection.drop()

    Collection.insert_many(reviews)
    
    client.close()

def get_reviews(movie_title):
    COLLECTION_NAME = movie_title

    client = MongoClient({MONGO_URI})
    database = client[DATABASE_NAME]
    Collection = database[COLLECTION_NAME]
    X=list()
    for x in Collection.find({}, {'_id':0}):
        
        result = {}
        result['review_text'] = x['review_text']
        result['review_star'] = x['review_star']
        X.append(result)
    
    client.close()
    return X

def insert_score(score):
    COLLECTION_NAME = 'score'

    client = MongoClient({MONGO_URI})
    database = client[DATABASE_NAME]
    Collection = database[COLLECTION_NAME]

    movie_title=score['title']
    Collection.delete_many({'title':movie_title})

    Collection.insert_one(score)
    
    client.close()

def get_score(movie_title):
    COLLECTION_NAME = 'score'

    client = MongoClient({MONGO_URI})
    database = client[DATABASE_NAME]
    Collection = database[COLLECTION_NAME]

    score = Collection.find_one({'title':movie_title}, {'_id':0})
    
    client.close()
    return score

def get_scores():
    COLLECTION_NAME = 'score'

    client = MongoClient({MONGO_URI})
    database = client[DATABASE_NAME]
    Collection = database[COLLECTION_NAME]

    scores = list()
    for x in Collection.find({}, {'_id':0}):
        scores.append(x)


    client.close()
    return scores

def get_infos():
    COLLECTION_NAME = 'info'

    client = MongoClient({MONGO_URI})
    database = client[DATABASE_NAME]
    Collection = database[COLLECTION_NAME]

    infos = list()
    for x in Collection.find({}, {'_id':0}):
        infos.append(x)


    client.close()
    return infos

def clean_both():
    COLLECTION_NAME = 'info'

    client = MongoClient({MONGO_URI})
    database = client[DATABASE_NAME]
    Collection = database[COLLECTION_NAME]
    Collection.drop()    
    client.close()
    COLLECTION_NAME = 'score'

    client = MongoClient({MONGO_URI})
    database = client[DATABASE_NAME]
    Collection = database[COLLECTION_NAME]
    Collection.drop()    
    client.close()



def to_data_csv():
    infos=list()
    infos=get_infos()

    labels=list()
    for x in infos[0].keys():
        labels.append(x)

    try:
        with open('summary.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=labels)
            writer.writeheader()
            for info in infos:
                writer.writerow(info)
    except IOError:
        print("I/O error")

