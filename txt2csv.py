import csv
import sys
import glob
import os.path

folder_name = "./classified_books/"

data_path = os.path.join(folder_name,'*txt')

files = glob.glob(data_path)

# all_texts = []
# all_names = []

# for file in files:
# 	file_name = file.replace(folder_name,'')
# 	all_names.append(file_name)
# 	with open(file) as f:
# 		data = f.readlines()
# 		data = ''.join(data)
# 		data = data.replace(',','')
# 		data = data.replace('"','')
# 		all_texts.append(data)


# with open('all_texts.csv', 'wb') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(all_names)
#     wr.writerow(all_texts)

all_texts = []

for file in files:
	file_name = file.replace(folder_name,'')
	with open(file) as f:
		data = f.readlines()
		data = ''.join(data)
		data = data.replace(',','')
		data = data.replace('"','')
		all_texts.append([data,file_name])


with open('all_texts.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(['content','file_name'])
    for elem in all_texts:
    	wr.writerow(elem)

