import os
import shutil
import json

input_dir = 'F:/ImageSet/hagrid_filter'
ref_dir = 'F:/ImageSet/hagrid_classified'

ref_classes = ['left_hand','right_hand']

subdir = os.listdir(input_dir)

output_dir = 'F:/ImageSet/hagrid_train'

json_file = 'updateCaption.json'
data = {}

if os.path.exists(json_file):
  # remove json file
  os.remove(json_file)

# create output dir
if not os.path.exists(output_dir):
  os.mkdir(output_dir)

sub_index = 0
total_sub = len(subdir)
# loop each subdir
for sub in subdir:
  sub_index+=1
  sub_dir = os.path.join(input_dir, sub)
  ref_sub_dir = os.path.join(ref_dir, sub)
  
  output_sub_dir = os.path.join(output_dir, sub.replace("_2048",""))
  # create subdir in output dir
  if not os.path.exists(output_sub_dir):
    os.mkdir(output_sub_dir)

  subdir_clear = os.path.join(sub_dir, 'clear')

  data[sub] = {
    'processed':[],
    'missing':[]
  }
  
  file_index = 0
  # because it has .jpg and .txt, divide 2
  total_image = len(os.listdir(subdir_clear))/2
  # loop each file in subdir_clear
  for file in os.listdir(subdir_clear):
    # skip when ext not eq to .txt
    if not file.endswith('.txt'):
      continue
    file_index+=1
    # get file ext
    file_name, file_ext = os.path.splitext(file)
    
    image_class = ''
    for ref_class in ref_classes:
      ref_class_dir = os.path.join(ref_sub_dir,ref_class)
      ref_image = os.path.join(ref_class_dir,f'{file_name}.jpg')
      if os.path.exists(ref_image):
        image_class = ref_class
        break
    # print(f'image_class: {image_class}')
    if image_class == '':
      data[sub]['missing'].append({
        'reason':'classified image not found',
        'file_name':file_name
      })
      print(f'sub_index: {sub_index}/{total_sub} file_index: {file_index}/{total_image} file: {file_name} skipped due to classified image not found')
      continue
    # read txt file
    with open(os.path.join(subdir_clear, file), 'r') as f:
      content = f.read()
      new_content = content
      # print(content)
      replaced = False
      if 'right hand' in content and image_class == 'left_hand':
        # when classified class contradict with caption
        new_content = new_content.replace('right hand','left hand')
        replaced = True
      if 'left hand' in content and image_class == 'right_hand':
        # when classified class contradict with caption
        new_content = new_content.replace('left hand','right hand')
        replaced = True

      # add hand classifier. raw photo to avoid affecting aestheic too much from raw photo
      new_content = f'{image_class}, {new_content}, raw photo'
      
      # copy image to output dir
      shutil.copy(os.path.join(subdir_clear, f'{file_name}.jpg'), output_sub_dir)
      # write new content to txt file
      with open(os.path.join(output_sub_dir, f'{file_name}.txt'), 'w') as new_f:
        # save new_content to txt file
        new_f.write(new_content)
        new_f.close()

      data[sub]['processed'].append({
        'file_name':file_name,
        'classified':image_class,
        'replaced':replaced,
        'new_caption':new_content,
        'before_caption':content,
      })
      f.close()
      print(f'sub_index: {sub_index}/{total_sub} file_index: {file_index}/{total_image} file: {file_name} processed.')
  #     break
  # break

with open(json_file, 'w', encoding='utf-8') as f:
  json.dump(data, f)