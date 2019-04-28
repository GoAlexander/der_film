import datetime 
import time
import concurrent.futures
import json

from selenium import webdriver 
import pandas as pd

#TODO:
# try headless moder (no gui)

#df=pd.read_csv('2.csv')
df=pd.read_csv('2_test.csv')
#CHROME_PATH = r'C:\\chromedriver.exe'
CHROME_PATH = r'C:\Users\Alexander\Downloads\chromedriver.exe'
NUMBER_OF_THREADS = 3
SCROLL_PAUSE_TIME = 30

# TODO: Is DataFrame thread safe?!
#data = pd.DataFrame(columns=['film', 'user', 'rating'])
data = []

def scraper_worker(pan):
    driver = webdriver.Chrome(CHROME_PATH)
    now = datetime.datetime.now() 
    driver.get(pan)  

    # Get scroll height 
    last_height = driver.execute_script("return document.body.scrollHeight") 

    while True: 
        # Scroll down to bottom 
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") 

        # Wait to load page 
        time.sleep(SCROLL_PAUSE_TIME) 

        # Calculate new scroll height and compare with last scroll height 
        new_height = driver.execute_script("return document.body.scrollHeight") 
        if new_height == last_height:
            break 
        last_height = new_height

    film=driver.find_element_by_xpath("//div[@id='content_block']/table/tbody/tr/td/div/ul/li[3]/h1/a") 
    time.sleep(0.001) 
    #films.append(film.text) #old

    time.sleep(0.001) 
    rows=driver.find_element_by_xpath("//table[@id='rating_list']/tbody") 
    time.sleep(0.001) 
    ds8s=rows.find_elements_by_xpath("//tr/td[3]/div/table/tbody/tr/td")
    # for ds8 in ds8s: 
    #     #res.append(ds8.text) #old
    #     time.sleep(0.001) 
    
    # ds9s=rows.find_elements_by_xpath("//tr/td[2]/div/p/a")
    # for ds9 in ds9s:
    #     #people.append(ds9.text)  #old
    #     #user = ds9.text
    #     #data = data.append(pd.DataFrame([[film.text, ds8.text, ds9.text]], columns=data.columns))
    #     data.append({"film": film.text, "rating": ds8.text, "user": ds9.text})
    #     print(data)
    #     #data.to_csv("kinopoisk_collector.txt", sep='\t', encoding='utf-8') #tmp
    #     time.sleep(0.001)
    
    # test:
    ds9s=rows.find_elements_by_xpath("//tr/td[2]/div/p/a")
    for ds8, ds9 in zip(ds8s, ds9s):
        film_str = film.text
        rating_str = ds8.text
        user_str = str(ds9.text).encode("utf-8")
        data.append({"film": film_str, "rating": rating_str, "user": user_str})
        time.sleep(0.001)    

    print(data.encode("utf-8")) # tmp
    then = datetime.datetime.now() 
    print(then-now)
    driver.close() #close webdriver process


pans=df['html']
with concurrent.futures.ThreadPoolExecutor(max_workers=NUMBER_OF_THREADS) as executor:
    features = {executor.submit(scraper_worker, pan): pan for pan in pans}
#concurrent.futures.wait(features, return_when=concurrent.futures.ALL_COMPLETED)
print("Final:")
print(data)

#data.to_csv("kinopoisk_collector.txt", sep='\t', encoding='utf-8')

# TODO: fix issues like:
# Traceback (most recent call last):
#   File "kinopoisk_collector.py", line 91, in <module>
#     json.dump(data, outfile)
#   File "C:\Users\Alexander\AppData\Local\Programs\Python\Python35\lib\json\__init__.py", line 178, in dump
#     for chunk in iterable:
#   File "C:\Users\Alexander\AppData\Local\Programs\Python\Python35\lib\json\encoder.py", line 427, in _iterencode
#     yield from _iterencode_list(o, _current_indent_level)
#   File "C:\Users\Alexander\AppData\Local\Programs\Python\Python35\lib\json\encoder.py", line 324, in _iterencode_list
#     yield from chunks
#   File "C:\Users\Alexander\AppData\Local\Programs\Python\Python35\lib\json\encoder.py", line 403, in _iterencode_dict
#     yield from chunks
#   File "C:\Users\Alexander\AppData\Local\Programs\Python\Python35\lib\json\encoder.py", line 436, in _iterencode
#     o = _default(o)
#   File "C:\Users\Alexander\AppData\Local\Programs\Python\Python35\lib\json\encoder.py", line 179, in default
#     raise TypeError(repr(o) + " is not JSON serializable")
# TypeError: b'kotovaeliz4' is not JSON serializable

# Fix should be like: https://stackoverflow.com/questions/24369666/typeerror-b1-is-not-json-serializable
with open('kinopoisk_collector.json', 'w') as outfile:
    json.dump(data, outfile)
