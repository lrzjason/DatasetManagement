import os
import json
import shutil

# provide input dir
# provide temp/below_list.json
# provide temp/upon_list.json
# provide output dir

# loop below_list and upon_list to get .jpg and .txt data
# copy .jpg and .txt to output dir

input_dir = 'F:/ImageSet/hagrid_train_all_classified'
ref_dir = 'F:/ImageSet/hagrid_train_all'
hands = ['left_hand','right_hand']

for subset in os.listdir(ref_dir):
    actual_left_hand_dir = os.path.join(input_dir, f'{subset}_left_hand')
    actual_right_hand_dir = os.path.join(input_dir, f'{subset}_right_hand')
    print(actual_left_hand_dir)
    print(actual_right_hand_dir)

    # get file name list with .txt without ext
    left_hand_file_list = [file_name.split('.')[0] for file_name in os.listdir(actual_left_hand_dir) if file_name.endswith('.txt')]
    right_hand_file_list = [file_name.split('.')[0] for file_name in os.listdir(actual_right_hand_dir) if file_name.endswith('.txt')]
    print(len(left_hand_file_list))
    print(len(right_hand_file_list))

    left_hand_total_diff = abs(len(left_hand_file_list) - len(right_hand_file_list))
    print(left_hand_total_diff)
    right_hand_remove_list = []
    if left_hand_total_diff > 0:
        for i in range(left_hand_total_diff):
            right_hand_remove_list.append(right_hand_file_list[i])
    for file_name in right_hand_remove_list:
        # remove .txt,.jpg,.npz from right_hand_dir
        os.remove(os.path.join(actual_right_hand_dir, f'{file_name}.txt'))
        os.remove(os.path.join(actual_right_hand_dir, f'{file_name}.jpg'))
        os.remove(os.path.join(actual_right_hand_dir, f'{file_name}.npz'))
    # break