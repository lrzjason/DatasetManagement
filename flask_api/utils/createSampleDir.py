import os

input_dir = 'F:/ImageSet/hagrid_train_all_classified'
output_dir = 'F:/lora_training/hand_gesture_training/sample'


repeat_prefix = "10_"

# create output directory if it doesn't exist
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

subdir = os.listdir(input_dir)

for sub in subdir:
  no_repeat = sub.replace(repeat_prefix,"")
  output_sub_dir = os.path.join(output_dir, no_repeat)
  if not os.path.exists(output_sub_dir):
    os.makedirs(output_sub_dir)