import json

# Load the data from the JSON file
with open('temp/result.json') as f:
    data = json.load(f)

below_score = 17
upon_score = 26

below_count = 0
upon_cout = 0

below_list = []
upon_list = []

# loop data
for item in data:
    print(item)
    if item['score'] > upon_score:
        upon_cout += 1
        upon_list.append(item)
    if item['score'] < below_score:
        below_count += 1
        below_list.append(item)

# sort the list by score ascending
upon_list.sort(key=lambda x:x['score'])
below_list.sort(key=lambda x:x['score'])

upon_list.append({'name':'summary', 'score': upon_cout})
below_list.append({'name':'summary', 'score': below_count})

# save the below_list and upon_list to json file
with open('temp/below_list.json', 'w') as f:
    json.dump(below_list, f)
with open('temp/upon_list.json', 'w') as f:
    json.dump(upon_list, f)

# Calculate the sum of the scores
sum_of_scores = sum(item['score'] for item in data)

# Calculate the average score
average_score = sum_of_scores / len(data)

print('Average score:', average_score)