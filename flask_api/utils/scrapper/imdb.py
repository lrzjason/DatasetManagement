import os
import shutil
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen

start_page = 0
paggination = 71

mediaview_url = "https://www.imdb.com/"
base_url = "https://www.imdb.com/title/tt0092455/mediaindex/"

#create directory based on base_url
folder = base_url.replace("https://www.imdb.com/title/", "")
if not os.path.exists(folder):
	os.makedirs(folder)

folder = "./"+folder + "/"
print("Created Directory:", folder)

for x in range(start_page, paggination):
	if x > 0: url =base_url +"?page="+str(x)
	else: url = base_url
	print()
	print("Scrapping from:", url)
	
	htmldata = urlopen(url)
	soup = BeautifulSoup(htmldata, 'html.parser')
	images = soup.find_all(class_='media_index_thumb_list')
	links = images[0].find_all('a')
	print("Found:", len(links), "images")
	
	for index, link in enumerate(links):
		thumb = link['href']
		url = mediaview_url + thumb
		print(url)
		imagedata = urlopen(url)
		individual_soup = BeautifulSoup(imagedata, 'html.parser')
		found = individual_soup.find('img')['src']
		if found == None:
			print("No image found")
			continue
		print("Downloading "+found)
		file_name = folder + found.split('/')[-1]

		try:
			res = requests.get(found, stream = True)
			if res.status_code == 200:
				exists = os.path.isfile(file_name)
				g = 0
				while exists:
					#print("file exists:", file_name)
					g += 1
					file_name =folder+str(index)+"_"+str(g)+"_"+ found.split('/')[-1]
					exists = os.path.isfile(file_name)
					
				with open(file_name,'wb') as f:
					shutil.copyfileobj(res.raw, f) 
				saved = os.path.isfile(file_name)
				if not saved:
					print(">>>> ",saved)
				print('Image sucessfully Downloaded: ',file_name)
			else:
				print('Image Couldn\'t be retrieved')
		except Exception as e:
			print(e)
			pass