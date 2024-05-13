import easyocr
import re
def find_double_quoted_content(input_string):
    # This regex pattern finds all content within double quotes
    pattern = r'"(.*?)"'
    # Find all matches of the pattern
    results = re.findall(pattern, input_string)
    return results

reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

path = 'F:/ImageSet/text_generation_017_styled/wrong/0_41.png'

result = " ".join(reader.readtext(path, detail = 0))
print(result)

# prompt = 'blurry foreground with text "Dream big, work harder", poster design'
# result = " ".join(find_double_quoted_content(prompt))
# print(result)

# if 'Dream big' in result:
#     print('contains')