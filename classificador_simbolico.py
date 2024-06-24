def main():
  from tokenizer import tokenize
  import nltk
  from collections import Counter
  from spellchecker import SpellChecker
  import pickle

  nltk.download('punkt')
  nltk.download('averaged_perceptron_tagger')
  
  # -------------------------- Utils ----------------------------# 

  # This function tokenize a text removing blank spaces
  def tokenize_phrase(text):
    words = tokenize(text)
    words_txt = [word.txt for word in words if word.txt != '']
    return words_txt

  # Calculates mean square erros of a dataframe, returns the percentage of each classification
  def freq_mean_square(freq, freq_ai, freq_not_ai):
    sum_ai = 0
    sum_not_ai = 0
    n = len(freq.items())

    for key, value in freq.items():
      if key in freq_ai:
        sum_ai += (value - freq_ai[key])**2

    for key, value in freq.items():
      if key in freq_not_ai:
        sum_not_ai += (value - freq_not_ai[key])**2
    n = sum_ai + sum_not_ai

    # Closer to AI x Closer to Human
    return 1-sum_ai/n, 1-sum_not_ai/n

  # Calculates mean square erros of a list, returns the percentage of each classification
  def freq_mean_square_list(freq, freq_ai, freq_not_ai):
    sum_ai = 0
    sum_not_ai = 0
    n = len(freq)

    for i, value in enumerate(freq):
      sum_ai += (value - freq_ai[i])**2
    for i, value in enumerate(freq):
      sum_not_ai += (value - freq_not_ai[i])**2

    n = sum_ai + sum_not_ai

    # Closer to AI x Closer to Human
    return [1-sum_ai/n, 1-sum_not_ai/n]

  # -------------------------- 1. Repetição de tags ----------------------------# 
  
  # Description: Makes the parsing(tagging) of a certain text
  # Params: text to be tagged
  # Return: the text tokenized and tagged
  def words_by_tag(text):
    wordtokens = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(wordtokens)
    return tagged_words

  # Description: Apply tagging in all texts of a dataframe
  # Params: dataframe with texts
  # Return: dictionary with all texts tagged
  def get_tagged_texts(_df):
    # Create an empty dictionary to store tagging for each text
    tagged_texts = {}

    # Iterate over each row in the DataFrame
    for index, row in _df.iterrows():
      text_id = index  # Assuming index represents text IDs
      text = row['text']

      # Tagging
      tagged_text = words_by_tag(text)

      # Store the tagged texts in the dictionary
      tagged_texts[text_id] = tagged_text

    return tagged_texts

  # Description: Returns k word mean frequencies
  # Params: dictionary of Counters(frequencies), value k
  # Return: dictionary with the highest k frequencies of k words
  def mean_word_frequency(frequencies_dict, k=5):

    sum_freq = [0]*k
    n = len(frequencies_dict.items())

    for key, value in frequencies_dict.items():
      # sort frequencies of words
      sorted_freq = sorted(dict(value).values(), reverse=True)[:k]
      # Equalize list size with zeros
      sorted_freq += [0]*(k-len(sorted_freq))
      # sum k frequencies of all texts from frequencies_list
      sum_freq = [x + y for x, y in zip(sum_freq, sorted_freq)]

    mean_k_freq = [x/n for x in sum_freq]
    return mean_k_freq

  # Description: Get the frequencies only of the specified tags
  # Params: dictionary with tagged texts, dictionary of tags to be considered
  # Return: dictionary of Counters with all the frequencies of each word
  def frequency_by_tag(texts, tags):
    new_dict = {}
    for key, value in texts.items():
      # select only the words in the tags list
      new_list = [word[0] for word in value if word[1] in tags]
      # Counter count the frequency of each word in the list
      new_dict[key] = Counter(new_list)

    return new_dict

  # All tags considered
  tags = {'Adjective': ['JJ'], 'Noun': ['NN', 'NNP', 'NNPS'], 'Adverb': ['RB', 'RBR', 'RBS'], 'Verb': ['VBD','VBZ', 'VBG', 'VBP', 'VB']}

  # Description: Get the highest k frequencies by word for each dataframe
  # Params: dictionary with AI texts, dictionary with Human texts
  # Return: tuple of the highest k frequencies by word for each dataframe
  def get_frequencies_by_class(df_ai, df_not_ai):
    tagged_ai = get_tagged_texts(df_ai)
    tagged_not_ai = get_tagged_texts(df_not_ai)

    frequency_tags_ai = {}
    frequency_tags_not_ai = {}

    for key, value in tags.items():

      # AI
      frequencies_ai = frequency_by_tag(tagged_ai, value)
      frequencies_mean_ai = mean_word_frequency(frequencies_ai)

      # Store results to AI
      frequency_tags_ai[key] = frequencies_mean_ai

      # NOT AI
      frequencies_not_ai = frequency_by_tag(tagged_not_ai, value)
      frequencies_mean_not_ai = mean_word_frequency(frequencies_not_ai)

      # Store results to Not AI
      frequency_tags_not_ai[key] = frequencies_mean_not_ai

    return frequency_tags_ai, frequency_tags_not_ai
  
  # Import files
  with open('frequency_tags_ai.pickle', 'rb') as file:
    frequency_tags_ai = pickle.load(file)

  with open('frequency_tags_not_ai.pickle', 'rb') as file:
    frequency_tags_not_ai = pickle.load(file)

  def classify_by_tag_frequency(text):

    # frequency_tags_ai, frequency_tags_not_ai = get_frequencies_by_class(df_ai, df_not_ai)

    # Inicialization
    tagged_text = { 'text': words_by_tag(text) }
    m_square = []
    sum_ai = 0
    sum_not_ai = 0

    # iterate over the tags
    for key, value in tags.items():

      # Calculate frequency
      frequency_text = frequency_by_tag(tagged_text, value)
      frequency_text_ = mean_word_frequency(frequency_text)

      # Get mean square
      m_square = freq_mean_square_list(frequency_text_, frequency_tags_ai[key], frequency_tags_not_ai[key])

      # Sum the results for a tag
      sum_ai += m_square[0]
      sum_not_ai += m_square[1]


    n = len(tags.items())
    mean_ai = sum_ai/n
    mean_not_ai = sum_not_ai/n

    return mean_ai, mean_not_ai


# ----------------- Get the frequency of each punctuation ------------------- #
  def punctuation_freq(text):
    dot = 0
    question = 0
    exclamation = 0
    dot_comma = 0
    comma = 0
    line = 0
    underline = 0
    for char in text:
      if char == ';':
        dot_comma += 1
      if char == '.':
        dot += 1
      if char == ',':
        comma += 1
      if char == '?':
        question += 1
      if char == '!':
        exclamation += 1
      if char == '-':
        line += 1
      if char == '_':
        underline += 1
      sum = dot + question + exclamation + comma
    return {'.':dot,'?':question,'!':exclamation,',':comma,'-':line,'_':underline,';':dot_comma}

  def mean_punctuation_freq(_df):
    # Iterate over each row in the DataFrame
    k = 1
    mean_freq = {}
    for index, row in _df.iterrows():
        text_id = index  # Assuming index represents text IDs
        text = row['text']

        # dict with ponctuation and frequency
        freq = punctuation_freq(text)
        for key, value in freq.items():
          if key in mean_freq:
            mean_freq[key] += value
          else:
            mean_freq[key] = value

        k += 1
    total = 0
    for key, value in mean_freq.items():
      total+=value
    for key, value in mean_freq.items():
      mean_freq[key] = 100*(value/total)
    return mean_freq
  
  # Import files
  with open('ai_pont_freq.pickle', 'rb') as file:
     ai_pont_freq = pickle.load(file)

  with open('not_ai_pont_freq.pickle', 'rb') as file:
    not_ai_pont_freq = pickle.load(file)

  # Classify a text by the punctuation frequency mean
  def classify_by_punctuation(text):
    freq = punctuation_freq(text)
    return freq_mean_square(freq, ai_pont_freq, not_ai_pont_freq)

# ---------------- 3. Comprimento das palavras ---------------------- #

  # Gets the frequency of a certain item of a array and returns the dict of it
  def frequency_dictionary(arr):
    frequency_dict = {}
    for item in arr:
      if item in frequency_dict:
        frequency_dict[item] += 1
      else:
        frequency_dict[item] = 1

    return frequency_dict

  # Calculates the mean of word lengths
  def average_word_length(words):
      # Calculates length of each word
      word_lengths = [len(word) for word in words]

      # Calculates the mean length of words
      average = sum(word_lengths) / len(word_lengths)
      return average

  # Calculates and returns a dict of all frequencies by word
  def freq_of_words(words):
      # Calculates length of each word
      word_lengths = [len(word) for word in words]

      freq = frequency_dictionary(word_lengths) # key: comprimento | value: frequencia
      ordered_freq = dict(sorted(freq.items(), key=lambda item: item[1]))
      return ordered_freq

  def mean_word_freq(_df):
    # Iterate over each row in the DataFrame
    k = 1
    mean_freq = {}
    word_total = 0
    df_size = len(_df)

    for index, row in _df.iterrows():
        text_id = index  # Assuming index represents text IDs
        text = row['text']

        # Tokenize the text into words
        words = tokenize_phrase(text)
        comprimento_medio = average_word_length(words)
        freq = freq_of_words(words) # key: length | value: frequency
        word_total += len(words)

        for key, value in freq.items():
          if key in mean_freq:
            mean_freq[key] += value
          else:
            mean_freq[key] = value

        k +=1

    for key, value in mean_freq.items():
      mean_freq[key] = value/df_size
    return mean_freq
  
  # Import files
  with open('ai_word_freq.pickle', 'rb') as file:
     ai_word_freq = pickle.load(file)

  with open('not_ai_word_freq.pickle', 'rb') as file:
    not_ai_word_freq = pickle.load(file)

  def classify_by_word_len(text):
    # Tokenize
    words = tokenize_phrase(text)
    freq = freq_of_words(words) # get frequency

    return freq_mean_square(freq, ai_word_freq, not_ai_word_freq)

# ------------------ 4. Desvio gramatical ---------------- #

  spell = SpellChecker()
  # Calculates the mean mispelled words in the texts of a dataframe
  def mean_misspelled_words(_df):

    mean_freq = {}
    misspelled_total = 0
    k = 0
    for index, row in _df.iterrows():
        text_id = index  # Assuming index represents text IDs
        text = row['text']
        words = tokenize_phrase(text)
        misspelled_count = len(spell.unknown(words))
        misspelled_total += misspelled_count
        k += 1
    return misspelled_total/k

  # Import files
  with open('mispelled_not_ai.pickle', 'rb') as file:
     mispelled_not_ai = pickle.load(file)

  with open('mispelled_ai.pickle', 'rb') as file:
    mispelled_ai = pickle.load(file)

  def classify_by_mispelled_freq(text):
    # Tokenize
    words = tokenize_phrase(text)
    misspelled_count = len(spell.unknown(words))

    # Calculates the difference between the values
    mean_ai = abs(misspelled_count - mispelled_ai)
    mean_not_ai = abs(misspelled_count - mispelled_not_ai)

    total = mean_ai+mean_not_ai

    return (1-mean_ai/total, 1-mean_not_ai/total)


# ------------------- 5. Resultado final -------------------- #

  # Using all criteria it classifies a text
  def classify_all_criteria(text):
    res_ai = 0
    res_not_ai = 0

    criteria_1 = classify_by_tag_frequency(text)
    criteria_2 = classify_by_punctuation(text)
    criteria_3 = classify_by_word_len(text)
    criteria_4 = classify_by_mispelled_freq(text)

    for i,item in enumerate([criteria_1, criteria_2, criteria_3, criteria_4]):
      res_ai += item[0]
      res_not_ai += item[1]

    total = res_ai + res_not_ai
    return (res_ai/total, res_not_ai/total)

  text = input('Insira um texto para classificação: ')

  mean_ai, mean_not_ai = classify_all_criteria(text)
  print(f"\nSemelhança com IA: {mean_ai} \nSemelhança com Humano: {mean_not_ai}\n")
  if mean_ai > mean_not_ai:
    print("Seu texto foi escrito por uma Inteligência Artificial")
  elif mean_ai < mean_not_ai:
    print("Seu texto foi escrito por um humano")
  else:
    print("Hmm estamos indecisos, o sistema não soube identificar o autor")

if __name__ == "__main__":
  main()