import random

def randomly_capitalize_string(input_string, percentage=50):
    """
    Randomly capitalize letters in a string based on the specified percentage.
    
    Args:
        input_string (str): The input string to modify
        percentage (int): Percentage of letters to capitalize (0-100)
    
    Returns:
        str: The modified string with randomly capitalized letters
    
    Raises:
        ValueError: If percentage is not between 0 and 100
    """
    if not isinstance(percentage, (int, float)) or percentage < 0 or percentage > 100:
        raise ValueError("Percentage must be a number between 0 and 100")
    
    if not input_string:
        return input_string
    
    # Convert percentage to decimal (e.g., 50% -> 0.5)
    probability = percentage / 100.0
    
    # Create a list of characters to modify
    result = list(input_string)
    
    # Iterate through each character and randomly capitalize based on probability
    for i in range(len(result)):
        char = result[i]
        # Only process alphabetic characters
        if char.isalpha():
            # Randomly decide whether to capitalize this letter
            if random.random() < probability:
                result[i] = char.upper()
            else:
                result[i] = char.lower()
    
    return ''.join(result)


def get_keyboard_substitution(char):
    """
    Get a realistic keyboard substitution for a given character.
    Based on QWERTY keyboard layout and common typing errors.
    
    Args:
        char (str): The character to find a substitution for
    
    Returns:
        str: A realistic substitution character
    """
    # QWERTY keyboard layout substitutions (common typos)
    substitutions = {
        # Row 1: qwertyuiop
        'q': ['w', 'a', '1', '2'],
        'w': ['q', 'e', 's', '2', '3'],
        'e': ['w', 'r', 'd', '3', '4'],
        'r': ['e', 't', 'f', '4', '5'],
        't': ['r', 'y', 'g', '5', '6'],
        'y': ['t', 'u', 'h', '6', '7'],
        'u': ['y', 'i', 'j', '7', '8'],
        'i': ['u', 'o', 'k', '8', '9'],
        'o': ['i', 'p', 'l', '9', '0'],
        'p': ['o', 'l', '0'],
        
        # Row 2: asdfghjkl
        'a': ['q', 's', 'z'],
        's': ['a', 'd', 'z', 'x'],
        'd': ['s', 'f', 'x', 'c'],
        'f': ['d', 'g', 'c', 'v'],
        'g': ['f', 'h', 'v', 'b'],
        'h': ['g', 'j', 'b', 'n'],
        'j': ['h', 'k', 'n', 'm'],
        'k': ['j', 'l', 'm'],
        'l': ['k', 'o', 'p'],
        
        # Row 3: zxcvbnm
        'z': ['a', 's', 'x'],
        'x': ['z', 's', 'd', 'c'],
        'c': ['x', 'd', 'f', 'v'],
        'v': ['c', 'f', 'g', 'b'],
        'b': ['v', 'g', 'h', 'n'],
        'n': ['b', 'h', 'j', 'm'],
        'm': ['n', 'j', 'k'],
        
        # Numbers (common number row substitutions)
        '1': ['q', '2'],
        '2': ['q', 'w', '1', '3'],
        '3': ['w', 'e', '2', '4'],
        '4': ['e', 'r', '3', '5'],
        '5': ['r', 't', '4', '6'],
        '6': ['t', 'y', '5', '7'],
        '7': ['y', 'u', '6', '8'],
        '8': ['u', 'i', '7', '9'],
        '9': ['i', 'o', '8', '0'],
        '0': ['o', 'p', '9'],
        
        # Common punctuation substitutions
        '.': [',', '/', '>'],
        ',': ['.', '<', '/'],
        ';': ['l', "'", '/'],
        "'": [';', 'l', '/'],
        '/': [';', "'", '.'],
        '\\': ['|', '/', 'z'],
        '[': ['p', 'o', ']'],
        ']': ['[', 'p', '\\'],
        '-': ['=', '_', '0'],
        '=': ['-', '_', '+'],
        
        # Space and common additions
        ' ': ['', 'x', 'c', 'v'],  # Common space typos
    }
    
    char_lower = char.lower()
    if char_lower in substitutions:
        return random.choice(substitutions[char_lower])
    else:
        # For characters not in our map, return common nearby characters
        return random.choice(['a', 'e', 'i', 'o', 'u', 's', 't', 'n', 'r'])


def get_keyboard_addition(char):
    """
    Get a realistic keyboard addition for a given character.
    Based on QWERTY keyboard layout and includes the original character
    to allow for duplicate characters (e.g., 'll' in 'hello').
    
    Args:
        char (str): The character to find an addition for
    
    Returns:
        str: A realistic addition character (can be the same as original)
    """
    # QWERTY keyboard layout additions (includes original character)
    additions = {
        # Row 1: qwertyuiop
        'q': ['q', 'w', 'a', '1', '2'],
        'w': ['w', 'q', 'e', 's', '2', '3'],
        'e': ['e', 'w', 'r', 'd', '3', '4'],
        'r': ['r', 'e', 't', 'f', '4', '5'],
        't': ['t', 'r', 'y', 'g', '5', '6'],
        'y': ['y', 't', 'u', 'h', '6', '7'],
        'u': ['u', 'y', 'i', 'j', '7', '8'],
        'i': ['i', 'u', 'o', 'k', '8', '9'],
        'o': ['o', 'i', 'p', 'l', '9', '0'],
        'p': ['p', 'o', 'l', '0'],
        
        # Row 2: asdfghjkl
        'a': ['a', 'q', 's', 'z'],
        's': ['s', 'a', 'd', 'z', 'x'],
        'd': ['d', 's', 'f', 'x', 'c'],
        'f': ['f', 'd', 'g', 'c', 'v'],
        'g': ['g', 'f', 'h', 'v', 'b'],
        'h': ['h', 'g', 'j', 'b', 'n'],
        'j': ['j', 'h', 'k', 'n', 'm'],
        'k': ['k', 'j', 'l', 'm'],
        'l': ['l', 'k', 'o', 'p'],
        
        # Row 3: zxcvbnm
        'z': ['z', 'a', 's', 'x'],
        'x': ['x', 'z', 's', 'd', 'c'],
        'c': ['c', 'x', 'd', 'f', 'v'],
        'v': ['v', 'c', 'f', 'g', 'b'],
        'b': ['b', 'v', 'g', 'h', 'n'],
        'n': ['n', 'b', 'h', 'j', 'm'],
        'm': ['m', 'n', 'j', 'k'],
        
        # Numbers (common number row additions)
        '1': ['1', 'q', '2'],
        '2': ['2', 'q', 'w', '1', '3'],
        '3': ['3', 'w', 'e', '2', '4'],
        '4': ['4', 'e', 'r', '3', '5'],
        '5': ['5', 'r', 't', '4', '6'],
        '6': ['6', 't', 'y', '5', '7'],
        '7': ['7', 'y', 'u', '6', '8'],
        '8': ['8', 'u', 'i', '7', '9'],
        '9': ['9', 'i', 'o', '8', '0'],
        '0': ['0', 'o', 'p', '9'],
        
        # Common punctuation additions
        '.': ['.', ',', '/', '>'],
        ',': [',', '.', '<', '/'],
        ';': [';', 'l', "'", '/'],
        "'": ["'", ';', 'l', '/'],
        '/': ['/', ';', "'", '.'],
        '\\': ['\\', '|', '/', 'z'],
        '[': ['[', 'p', 'o', ']'],
        ']': [']', '[', 'p', '\\'],
        '-': ['-', '=', '_', '0'],
        '=': ['=', '-', '_', '+'],
        
        # Space and common additions
        ' ': [' ', 'x', 'c', 'v'],  # Common space typos
    }
    
    char_lower = char.lower()
    if char_lower in additions:
        return random.choice(additions[char_lower])
    else:
        # For characters not in our map, return common nearby characters including original
        return random.choice([char_lower, 'a', 'e', 'i', 'o', 'u', 's', 't', 'n', 'r'])


def introduce_typos(input_string, flip_rate=5, drop_rate=3, add_rate=2, substitute_rate=4):
    """
    Introduce common typos into a string with configurable rates.
    
    The modifications are applied in order: drops, substitutions, flips, additions
    to avoid compounding effects (e.g., dropping 100% eliminates need for further processing).
    
    Args:
        input_string (str): The input string to modify
        flip_rate (int): Percentage of adjacent letter pairs to flip (0-100)
        drop_rate (int): Percentage of characters to drop (0-100)
        add_rate (int): Percentage of positions to add random characters (0-100)
        substitute_rate (int): Percentage of characters to substitute with keyboard neighbors (0-100)
    
    Returns:
        str: The modified string with introduced typos
    
    Raises:
        ValueError: If any rate is not between 0 and 100
    """
    # Validate input parameters
    for rate, name in [(flip_rate, "flip_rate"), (drop_rate, "drop_rate"), 
                       (add_rate, "add_rate"), (substitute_rate, "substitute_rate")]:
        if not isinstance(rate, (int, float)) or rate < 0 or rate > 100:
            raise ValueError(f"{name} must be a number between 0 and 100")
    
    if not input_string:
        return input_string
    
    # Convert rates to probabilities
    flip_prob = flip_rate / 100.0
    drop_prob = drop_rate / 100.0
    add_prob = add_rate / 100.0
    substitute_prob = substitute_rate / 100.0
    
    # Step 1: Drop characters (highest priority to avoid compounding)
    result = list(input_string)
    if drop_prob > 0:
        # Process from end to beginning to avoid index issues
        for i in range(len(result) - 1, -1, -1):
            if random.random() < drop_prob:
                result.pop(i)
    
    # If all characters were dropped, return empty string
    if not result:
        return ""
    
    # Step 2: Substitute characters with keyboard neighbors
    if substitute_prob > 0:
        for i in range(len(result)):
            if random.random() < substitute_prob:
                substitution = get_keyboard_substitution(result[i])
                if substitution:  # Don't substitute if we get an empty string
                    result[i] = substitution
    
    # Step 3: Flip adjacent letters
    if flip_prob > 0 and len(result) > 1:
        # Process pairs of adjacent characters
        for i in range(len(result) - 1):
            if random.random() < flip_prob:
                # Swap adjacent characters
                result[i], result[i + 1] = result[i + 1], result[i]
    
    # Step 4: Add random characters (keyboard neighbors of existing chars)
    if add_prob > 0:
        # Process from end to beginning to avoid index issues
        for i in range(len(result), -1, -1):
            if random.random() < add_prob:
                # Choose a character to base the addition on
                if i < len(result):
                    base_char = result[i]
                elif i > 0:
                    base_char = result[i - 1]
                else:
                    base_char = 'a'  # Default if no context
                
                # Get a keyboard neighbor for the base character (including original for duplicates)
                random_char = get_keyboard_addition(base_char)
                if random_char:  # Only add if we get a valid character
                    result.insert(i, random_char)
    
    return ''.join(result)


def introduce_typos_per_word(input_string, typos_per_word=1.0, typo_types=None):
    """
    Introduce typos into a string with a specified rate per word.
    
    Args:
        input_string (str): The input string to modify
        typos_per_word (float): Number of typos per word. Values < 1 indicate probability 
                               (e.g., 0.5 = 50% chance of typo per word)
        typo_types (set): Set of typo types to use. Default is all types.
                         Options: {'substitute_rate', 'flip_rate', 'drop_rate', 'add_rate'}
    
    Returns:
        str: The modified string with introduced typos
    
    Raises:
        ValueError: If typos_per_word is negative or typo_types is invalid
    """
    if not input_string:
        return input_string
    
    if typos_per_word < 0:
        raise ValueError("typos_per_word must be non-negative")
    
    # Default typo types if none specified
    if typo_types is None:
        typo_types = {'substitute_rate', 'flip_rate', 'drop_rate', 'add_rate'}
    
    # Validate typo types
    valid_types = {'substitute_rate', 'flip_rate', 'drop_rate', 'add_rate'}
    if not typo_types.issubset(valid_types):
        raise ValueError(f"Invalid typo types. Must be subset of {valid_types}")
    
    if not typo_types:
        return input_string  # No typo types specified, return original
    
    # Split into words (preserve whitespace)
    import re
    words = re.split(r'(\s+)', input_string)
    result_words = []
    
    for word in words:
        if not word.strip():  # Skip pure whitespace
            result_words.append(word)
            continue
        
        modified_word = word
        
        # Determine number of typos for this word
        if typos_per_word < 1:
            # Probability-based: chance of getting exactly one typo
            num_typos = 1 if random.random() < typos_per_word else 0
        else:
            # Integer-based: get floor(typos_per_word) + chance of extra
            base_typos = int(typos_per_word)
            extra_chance = typos_per_word - base_typos
            num_typos = base_typos + (1 if random.random() < extra_chance else 0)
        
        # Apply typos
        for _ in range(num_typos):
            if len(modified_word) < 2:  # Need at least 2 chars for most typos
                break
            
            # Randomly select typo type
            typo_type = random.choice(list(typo_types))
            
            if typo_type == 'substitute_rate':
                # Substitute a random character (skip spaces)
                if modified_word:
                    char_list = list(modified_word)
                    # Find non-space characters
                    non_space_indices = [i for i, char in enumerate(char_list) if char != ' ']
                    if non_space_indices:
                        idx = random.choice(non_space_indices)
                        substitution = get_keyboard_substitution(char_list[idx])
                        if substitution:
                            char_list[idx] = substitution
                        modified_word = ''.join(char_list)
            
            elif typo_type == 'flip_rate':
                # Flip adjacent characters (skip spaces)
                if len(modified_word) >= 2:
                    char_list = list(modified_word)
                    # Find pairs of adjacent non-space characters
                    valid_pairs = []
                    for i in range(len(char_list) - 1):
                        if char_list[i] != ' ' and char_list[i + 1] != ' ':
                            valid_pairs.append(i)
                    
                    if valid_pairs:
                        idx = random.choice(valid_pairs)
                        char_list[idx], char_list[idx + 1] = char_list[idx + 1], char_list[idx]
                        modified_word = ''.join(char_list)
            
            elif typo_type == 'drop_rate':
                # Drop a random character (skip spaces)
                if len(modified_word) >= 2:
                    char_list = list(modified_word)
                    # Find non-space characters
                    non_space_indices = [i for i, char in enumerate(char_list) if char != ' ']
                    if non_space_indices:
                        idx = random.choice(non_space_indices)
                        char_list.pop(idx)
                        modified_word = ''.join(char_list)
            
            elif typo_type == 'add_rate':
                # Add a random character (skip spaces)
                char_list = list(modified_word)
                if char_list:
                    # Find positions next to non-space characters
                    valid_positions = []
                    for i in range(len(char_list) + 1):
                        # Check if position is adjacent to a non-space character
                        if (i > 0 and char_list[i-1] != ' ') or (i < len(char_list) and char_list[i] != ' '):
                            valid_positions.append(i)
                    
                    if valid_positions:
                        idx = random.choice(valid_positions)
                        # Choose character to insert (based on nearby character or default)
                        if idx < len(char_list):
                            base_char = char_list[idx]
                        elif idx > 0:
                            base_char = char_list[idx - 1]
                        else:
                            base_char = 'a'
                        
                        new_char = get_keyboard_addition(base_char)
                        if new_char:
                            char_list.insert(idx, new_char)
                        modified_word = ''.join(char_list)
        
        result_words.append(modified_word)
    
    return ''.join(result_words)


# Example usage and testing
if __name__ == "__main__":
    # Test the function with different percentages
    test_string = "hello world this is a test string"
    
    print("Original string:", test_string)
    print()
    
    # Test with different percentages
    percentages = [0, 25, 50, 75, 100]
    
    for pct in percentages:
        result = randomly_capitalize_string(test_string, pct)
        print(f"{pct}% capitalization: {result}")
    
    print()
    
    # Test with a few random runs to show the randomness
    print("Random runs with 50% capitalization:")
    for i in range(3):
        result = randomly_capitalize_string(test_string, 50)
        print(f"Run {i+1}: {result}")
    
    print("\n" + "="*50)
    print("TYPO INTRODUCTION TESTS")
    print("="*50)
    
    # Test typo introduction with different rates
    print(f"Original: {test_string}")
    print()
    
    # Test with low rates
    result = introduce_typos(test_string, flip_rate=10, drop_rate=5, add_rate=3, substitute_rate=8)
    print(f"Low rates (10% flip, 5% drop, 3% add, 8% substitute): {result}")
    
    # Test with medium rates
    result = introduce_typos(test_string, flip_rate=20, drop_rate=10, add_rate=8, substitute_rate=15)
    print(f"Medium rates (20% flip, 10% drop, 8% add, 15% substitute): {result}")
    
    # Test with high rates
    result = introduce_typos(test_string, flip_rate=40, drop_rate=20, add_rate=15, substitute_rate=25)
    print(f"High rates (40% flip, 20% drop, 15% add, 25% substitute): {result}")
    
    # Test extreme case (100% drop rate)
    result = introduce_typos(test_string, flip_rate=50, drop_rate=100, add_rate=30, substitute_rate=20)
    print(f"100% drop rate: '{result}'")
    
    print()
    
    # Test multiple random runs
    print("Random typo runs (15% flip, 8% drop, 5% add, 12% substitute):")
    for i in range(3):
        result = introduce_typos(test_string, flip_rate=15, drop_rate=8, add_rate=5, substitute_rate=12)
        print(f"Run {i+1}: {result}")
    
    print("\n" + "="*50)
    print("KEYBOARD SUBSTITUTION EXAMPLES")
    print("="*50)
    
    # Test keyboard substitutions
    test_chars = "hello world"
    print(f"Original: {test_chars}")
    print("Common substitutions:")
    for char in test_chars:
        if char.isalpha():
            substitution = get_keyboard_substitution(char)
            print(f"  '{char}' → '{substitution}'")
    
    print("\n" + "="*50)
    print("INDIVIDUAL TYPO TYPE TESTS")
    print("="*50)
    
    # Test each typo type individually
    test_string = "hello world"
    print(f"Original string: '{test_string}'")
    print()
    
    # Test 1: Character Drops Only
    print("1. CHARACTER DROPS (drop_rate=30%, others=0%):")
    for i in range(3):
        result = introduce_typos(test_string, flip_rate=0, drop_rate=30, add_rate=0, substitute_rate=0)
        print(f"   Run {i+1}: '{result}'")
    print()
    
    # Test 2: Character Substitutions Only
    print("2. CHARACTER SUBSTITUTIONS (substitute_rate=40%, others=0%):")
    for i in range(3):
        result = introduce_typos(test_string, flip_rate=0, drop_rate=0, add_rate=0, substitute_rate=40)
        print(f"   Run {i+1}: '{result}'")
    print()
    
    # Test 3: Letter Flips Only
    print("3. LETTER FLIPS (flip_rate=25%, others=0%):")
    for i in range(3):
        result = introduce_typos(test_string, flip_rate=25, drop_rate=0, add_rate=0, substitute_rate=0)
        print(f"   Run {i+1}: '{result}'")
    print()
    
    # Test 4: Character Additions Only
    print("4. CHARACTER ADDITIONS (add_rate=20%, others=0%):")
    for i in range(3):
        result = introduce_typos(test_string, flip_rate=0, drop_rate=0, add_rate=20, substitute_rate=0)
        print(f"   Run {i+1}: '{result}'")
    print()
    
    # Test 5: Show the ordering effect
    print("5. ORDERING EFFECT DEMONSTRATION:")
    print("   Original: 'hello world'")
    print("   With 100% drop rate (should eliminate all other effects):")
    result = introduce_typos(test_string, flip_rate=100, drop_rate=100, add_rate=100, substitute_rate=100)
    print(f"   Result: '{result}'")
    print("   Note: All characters dropped, so no flips/substitutions/additions occur")
    print()
    
    # Test 6: Show realistic combinations
    print("6. REALISTIC COMBINATIONS:")
    print("   Low rates (realistic typing errors):")
    for i in range(2):
        result = introduce_typos(test_string, flip_rate=8, drop_rate=5, add_rate=3, substitute_rate=12)
        print(f"   Run {i+1}: '{result}'")
    
    print("   Medium rates (more noticeable errors):")
    for i in range(2):
        result = introduce_typos(test_string, flip_rate=20, drop_rate=15, add_rate=10, substitute_rate=25)
        print(f"   Run {i+1}: '{result}'") 