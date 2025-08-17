def convert_word_to_number(word):
    word_dict = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
        'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
    }
    
    if isinstance(word, (int, float)):
        return word
    
    word = word.lower().strip()
    
    # Check if it's a simple word in our dictionary
    if word in word_dict:
        return word_dict[word]
    
    # Handle compound words (like twenty-one or twenty one)
    if '-' in word:
        parts = word.split('-')
    else:
        parts = word.split()
    
    # Simple case for two-part numbers (e.g., twenty-one)
    if len(parts) == 2:
        if parts[0] in word_dict and parts[1] in word_dict:
            return word_dict[parts[0]] + word_dict[parts[1]]
    
    # If not recognized, return original value or handle as needed
    try:
        # Try to convert if it's already a numeric string
        return float(word) if '.' in word else int(word)
    except ValueError:
        return word  # or return some default value

# Apply the function to the DataFrame column
d.experience = d.experience.apply(convert_word_to_number)