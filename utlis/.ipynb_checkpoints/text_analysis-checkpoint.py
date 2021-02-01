import nltk
# nltk.download()
import re 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import sent_tokenize, word_tokenize, FreqDist
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from math import ceil
from collections import Counter
from wordcloud import STOPWORDS

stop_words = set(stopwords.words('english'))


class FeatureEng():
    
    # Basic clean function
    def clean_text(text):
        '''Input: Uncleaned text
           Returns: cleaned text'''
        text = re.sub("\'", "", text) #Removing backslashes
        text = re.sub("[^a-zA-Z]"," ",text) #Keeping english alphabets only
        text = ' '.join(text.split()) #Striping extra white-spaces
        text = text.lower() #Normalizing text
        return text
    
    def remove_stopwords(text):
        '''Input: string
           Returns: string without stopwords'''
        no_stopword_text = [w for w in text.split() if not w in stop_words]
        return ' '.join(no_stopword_text)
    
    def stem_corpus(text):
        '''Input: string
           Returns: string after performing stemming'''
        stemmer = SnowballStemmer("english")
        return stemmer.stem(text)
    
    #Normalization function
    def normalization_word(s):
        '''Function: Normalization over no. of words
           Returns: number of words.'''
        words = nltk.Text(word_tokenize(((s))))
        return len(words)

    def normalization_sentence(s):
        '''Function: Normalization by no. of Sentences
           Returns : number of sentences.'''
        sentences = nltk.Text(sent_tokenize(s))
        return len(sentences)
    
    def normalization_character(s):
        '''Function: Normalization by no. of Characters
           Returns: number of characters.'''
        return len(s)

    def cleaning(s):
        '''Function: basic cleaning function
           Input: Take a string. 
           Returns: a string with only lowercase letters & the space between words.'''
        plain_string = ""
        for x in s:
            x = x.lower()
            if (('a' <= x and x <= 'z') or x == ' '):
                plain_string += x
            elif x == '\'':  # any apostrophes(') are replaced by a space
                plain_string += ' '
        while '  ' in plain_string:  # if any multiple spaces detected, they are replaced by a single space
            plain_string = plain_string.replace('  ', ' ')
        return plain_string

    # Average sentence length (in char)
    def length_character(s):
        '''Input: Takes a string
           Returns: an int (average sentence length in characters).'''
        return len(s) / FeatureEng.normalization_sentence(s)
    
    # Average sentence length (in words)
    def length_sentence(s):
        '''Input: Takes a string
           Returns: an int (average sentence length in words).'''
        return len(s.split()) / FeatureEng.normalization_sentence(s)

    # Average characters per word
    def length_word(s):
        '''Input: Takes a string
           Returns: an int (average characters per word). Excludes punctuations.'''
        return len(s.split()) / FeatureEng.normalization_word(s)

    # For capturing Punctuation density 
    def density_coma(s):
        '''Input: Takes a string
           Returns: the ratio of punctuations to characters.'''
        counter = 0
        for x in s:
            if x == ',':
                counter += 1
        return counter / FeatureEng.normalization_character(s)

    def density_point(s):
        '''Input: Takes a string
           Returns: the ratio of periods(.) to characters.'''
        counter = 0
        for x in s:
            if x == '.':
                counter += 1
        return counter / FeatureEng.normalization_character(s)

    def density_colon(s):
        '''Input: Takes a string 
           Returns: the ratio of colons(:) to characters.'''
        counter = 0
        for x in s:
            if x == ':':
                counter += 1
        return counter / FeatureEng.normalization_character(s)

    def density_semicolon(s):
        '''Input: Takes a string
           Returns: the ratio of semicolons(;) to characters.'''
        counter = 0
        for x in s:
            if x == ';':
                counter += 1
        return counter / FeatureEng.normalization_character(s)
    
    def density_interro(s):
        '''Input: Takes a string
           Returns: the ratio of question marks(?) to characters.'''
        counter = 0
        for x in s:
            if x == '?':
                counter += 1
        return counter / FeatureEng.normalization_character(s)

    def density_expl(s):
        '''Input: Takes a string
           Returns: the ratio of exclamation points(!) to characters.'''
        counter = 0
        for x in s:
            if x == '!':
                counter += 1
        return counter / FeatureEng.normalization_character(s)

    # For calculating percentage of unique words per sentence ( to get richness of vocab)
    def vocabulary_sentence(s):
        '''Input: Takes a string
           Returns: the ratio of different words to all words.'''
        s = nltk.Text(sent_tokenize(s))
        vocabulary_list = []
        for c in s:
            if FeatureEng.normalization_word(c) != 0:
                vacabulary_count_sentence = len({x.lower() for x in word_tokenize(FeatureEng.cleaning(c))})
                vocabulary_list.append(vacabulary_count_sentence / FeatureEng.normalization_word(c))
        return np.mean(vocabulary_list)

    # Stopword percentage
    def density_stopword(s):
        '''Input: Takes a string
           Returns: the ratio of stopwords to all words.'''
        cs = 0
        for x in nltk.Text(word_tokenize(s)):
            if x in STOPWORDS:
                cs += 1
        return cs/FeatureEng.normalization_word(s)
    
    # Noun Density
    def density_noun(s):
        '''Input: Takes a string
           Returns: the ratio of nouns to all words.'''
        l = []
        for x in nltk.pos_tag(word_tokenize(s)):
            if x[1][0:2] == 'NN': # all noun tags start with NN
                l.append(x)
        return len(l)/FeatureEng.normalization_word(s)

    # Verb Density
    def density_verb(s):
        '''Input: Takes a string
           Returns: the ratio of verbs to all words.'''
        l = []
        for x in nltk.pos_tag(word_tokenize(s)):
            if x[1][0:2] == 'VB': # all verb tags start with VB
                l.append(x)
        return len(l)/FeatureEng.normalization_word(s)

    # Adjective Density
    def density_adjective(s):
        '''Input: Takes a string
           Returns: the ratio of adjectives to all words.'''
        l = []
        for x in nltk.pos_tag(word_tokenize(s)):
            if x[1][0:2] == 'JJ': # all adjective tags start with JJ
                l.append(x)
        return len(l) / FeatureEng.normalization_word(s)

    # Adjective to Noun Ratio
    def adjective_to_noun(s):
        '''Input: Takes a string
           Returns: the ratio of adjectives to nouns.'''
        return FeatureEng.density_adjective(s) / (FeatureEng.density_noun(s) + 0.5) # add 0.5 to avoid division by 0 error

    # Count of Emphases  used on Words or Phrases used
    def count_emph(s):
        '''Input: Takes a string
           Returns: the usage count of emphases using double quotes.'''
        emph_trig_words = 'word called the a their my his her for that those like of words'.split() 
        emph_count = 0
        s = s.lower()
        for word in emph_trig_words:
            emph_count += s.count('{} "'.format(word))
        return emph_count

    # Count of dialogue breaks used
    def count_dial_break(s):
        '''Input: Takes a string
           Returns: the count of dialogue breaks used.'''
        return s.count(", \"")
    
    # Dialogues used : to observe particularity/ style of an author
    def count_dblqt(s):
        '''Input: Takes a string
           Returns: the count of sets of double quotes'''
        return ceil(s.count('"')/2) # ceil function rounds up
    
    def count_dial(s):
        '''Input: Takes a string
           Returns: the count of dialogues used (if in double quotes but not an emphasis, then it is a dialogue.)'''
        return FeatureEng.count_dblqt(s) - FeatureEng.count_emph(s) - FeatureEng.count_dial_break(s)

    def break_to_dial_ratio(s):
        """Take sa string and returns the ratio of dialogue breaks to dialogues."""
        if not FeatureEng.count_dial(s):
            return 0
        return FeatureEng.count_dial_break(s) / FeatureEng.count_dial(s)

    # Observing count of soft words over masculine words used
    def count_fem(s):
        '''Input: Takes a string
           Returns: the count of feminine words.'''
        fem_words = 'she her woman herself girl women lady queen princess daughter madam madame wife'.split()
        fem_count = 0
        s = FeatureEng.cleaning(s)
        for word in s.split():
            if word in fem_words:
                fem_count += 1
        return fem_count
    
    def count_mas(s):
        '''Input: Takes a string
           Returns: the count of masculine words.'''
        mas_words = 'he his man mr himself boy men gentleman gentlemen king prince son sir husband'.split()
        mas_count = 0
        s = FeatureEng.cleaning(s)
        for word in s.split():
            if word in mas_words:
                mas_count += 1
        return mas_count

    def fem_to_mas_ratio(s):
        '''Input: Takes a string
           Returns: the ratio of feminine words to masculine words.'''
        fem_count = FeatureEng.count_fem(s)
        mas_count = FeatureEng.count_mas(s)
        if fem_count and not mas_count:
            fem_mas_ratio = 1
        elif not fem_count and not mas_count:
            fem_mas_ratio = 0
        else:
            fem_mas_ratio = fem_count / mas_count
        return fem_mas_ratio
    
    #For visualization 
    def freq_words(x, terms = 30):
        '''Input: A dataframe and no. of most frequent words (default is set to 30)
           Returns: plot with n number of most frequent words'''
        all_words = ' '.join([text for text in x]) 
        all_words = all_words.split() 
        fdist = nltk.FreqDist(all_words)
        words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())}) 
        # selecting top 20 most frequent words 
        d = words_df.nlargest(columns="count", n = terms) 
        # visualize words and frequencies
        plt.figure(figsize=(12,15)) 
        ax = sns.barplot(data=d, x= "count", y = "word") 
        ax.set(ylabel = 'Word') 
        return plt.show()
    
    def get_nrc_data():
        '''Function: Builds an emotion dictionary from the NRC emotion lexicon.'''
        nrc = "../data/emotion-lexicon-wordlevel-alphabetized.txt"
        count = 0
        emotion_dict = dict()
        with open(nrc,'r') as f:
            all_lines = list()
            for line in f:
                if count < 46:
                    count += 1
                    continue
                line = line.strip().split('\t')
                if int(line[2]) == 1:
                    if emotion_dict.get(line[0]):
                        emotion_dict[line[0]].append(line[1])
                    else:
                        emotion_dict[line[0]] = [line[1]]
        return emotion_dict

    def emotions(dataframe):
        '''Input:Takes a dataframe
           Return: a column per emotion with a count on the emotions used.'''
        df = dataframe.copy()
        emotions=['positive','anger','disgust','fear','negative','sadness','anticipation','joy','surprise','trust']
        for emotion in emotions:
            df[emotion]=0
        emotion_dic = FeatureEng.get_nrc_data()
        for i in range(len(df)):
            words = df['content'].iloc[i].split()
            n = len(words)
            for word in words:
                if word in emotion_dic:
                    for emotion in emotion_dic[word]:
                        column = emotion
                        try:
                            df[column].iloc[i] = df[column].iloc[i] + 1 / n
                        except:
                            pass
        return df

if  __name__  ==  "__main__" : 
    s = 'The text cleaning techniques we have seen so "far work very well in practice". Depending on the kind;  of texts you may encounter, it may be relevant to include more complex text cleaning steps. But keep in mind that the more steps we add, the longer the text cleaning will take. '
    func = [FeatureEng.count_emph(s), FeatureEng.count_dial(s), FeatureEng.count_dial_break(s),FeatureEng.break_to_dial_ratio(s),
            FeatureEng.count_fem(s), FeatureEng.count_mas(s),FeatureEng.fem_to_mas_ratio(s)]
    for i in func:
        print(i)
