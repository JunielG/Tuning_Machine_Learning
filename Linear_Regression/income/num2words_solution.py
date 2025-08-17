import pandas as pd
from num2words import num2words

# Create a mapping of words to numbers
max_num = 100  # Adjust based on your expected range
word_to_number = {num2words(i): i for i in range(max_num+1)}

# Apply the mapping
d.experience = d.experience.apply(lambda x: word_to_number.get(str(x).lower(), x))