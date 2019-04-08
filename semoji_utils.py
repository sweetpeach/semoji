# -*- coding: utf-8 -*-
"""
13 Mar 2018
To keep various shared code
"""

# Import statements
from __future__ import print_function, division
import sys
import re
from nltk.tokenize import casual_tokenize
# !pip install emoji_list
import emoji_list
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# !pip install seaborn
import seaborn as sns
font = {
        'weight': 'normal',
        'size': 30}
mpl.rc('font', **font)
import logging
import csv
import os

def get_logger():
    logger = logging.getLogger('Utils')
    logger.setLevel(logging.INFO)
    handler_count = 0
    _logger = logger
    while _logger is not None:
        handler_count += len(_logger.handlers)
        _logger = _logger.parent
    if handler_count == 0:
        logger.addHandler(logging.StreamHandler())
    return logger

def read_data(path, emotion_path=None, limit_data=-1):
    """Read tweet-emoji dataset, handling various different formats.

    Returned value is a tuple: (tweets, emojis, tweet IDs, emotions, headers)

    This handles these cases:
    - When the data has headers (by checking the first row whether it matches 'id', 'tweet', or 'emoji' at expected columns)
    - When the data has tweet ID as the first column
    - When the data contains emotion information
    - When the emotion file is separate
    """
    indices = []
    tweets = []
    labels = []
    emotions = []
    has_headers = False
    has_tweet_id = False
    has_emotion = False
    has_escaped_quote = False
    if not os.path.exists(path):
        raise ValueError('Data path "{}" does not exist'.format(path))
    with open(path, 'r') as infile:
        headers = None
        for line_idx, line in enumerate(infile.readlines()):
            row = line.strip().split('\t')
            if line_idx == 0:
                if row[0].lower() in ['', 'id', 'tweet'] or row[1].lower() in ['tweet', 'emoji']:
                    has_headers = True
                    headers = row
                    continue
            if len(row) == 3: # With ID at first column: tweet_id tweet eoji
                idx, tweet, label = row
                indices.append(idx)
                has_tweet_id = True
            elif len(row) == 2:  # Original Barbieri format: tweet eoji
                tweet, label = row
            else: # Merged with emotion, assume always have tweet ID as first column: tweet_id tweet emo1 ... emoN eoji
                idx = row[0]
                indices.append(idx)
                tweet, label = row[1], row[-1]
                emotion = [float(v) for v in row[2:-1]]
                emotions.append(emotion)
                has_emotion = True
                has_tweet_id = True
            if tweet[0] == tweet[-1] == '"' and '""' in tweet[1:-1] and ' " ' not in tweet[1:-1]:
                # Due to the use of csv module at some point, sometimes the double-quotes are escaped
                tweet = tweet[1:-1].replace('""', '"')
                has_escaped_quote = True
            tweets.append(tweet)
            labels.append(label)
            if limit_data > 0 and len(tweets) == limit_data:
                break
    if emotion_path is not None:
        # Read separate emotion file
        # First line is header: ID tweet emo1 emo2 ... emoN
        emotions = []
        add_indices = bool(len(indices) == 0)
        with open(emotion_path, 'r') as infile:
            reader = csv.reader(infile, delimiter='\t')
            headers = next(reader)
            # headers is now ['ID', 'tweet', 'emo1', 'emo2', ..., 'emoN']
            headers.append('emoji')
            for row in reader:
                if add_indices:
                    indices.append(row[0])
                emotions.append([float(v) for v in row[2:]])
                if limit_data > 0 and len(emotions) == limit_data:
                    break
    get_logger().info('Data format: [{}headers][{}ID][{}emotion][{}escaped]'.format(
                                      '+' if has_headers else '-',
                                      '+' if has_tweet_id else '-',
                                      '+' if has_emotion else '-',
                                      '+' if has_escaped_quote else '-'))
    return tweets, labels, indices, emotions, headers

def draw_radar(data, labels=None, title='Radar chart', title_y=1.05, fig=None, subplot=None, rlim=(0,1)):
    """Draw emotion radar"""
    if labels is None:
        labels = ['']*len(data)
    angles = np.linspace(0, 2*np.pi, len(data), endpoint=False)
    data=np.concatenate((data,[data[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    if fig is None:
        fig=plt.figure()
    if subplot is None:
        subplot = 111
    if type(subplot) is int:
        subplot = map(int, str(subplot))
    ax = fig.add_subplot(*subplot, polar=True)
    ax.set_rlim(*rlim)
    ax.plot(angles, data, 'o-', linewidth=2)
    ax.fill(angles, data, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.set_title(title, y=title_y)
    return fig, ax

EMOJIS = [
    ('eoji1f602','1f602',['😂'],'Face with tears of joy','eoji1f602'),         #0
    ('eoji2764','2764',['❤️','❤'],'Red Heart','eoji2764'),                     #1
    ('eoji1f60d','1f60d',['😍'],'Smiling face with heart-eyes','eoji2764'),    #2
    ('eoji1f525','1f525',['🔥'],'Fire','eoji1f525'),                           #3
    ('eoji1f4af','1f4af',['💯'],'100','eoji1f4af'),                            #4
    ('eoji1f60a','1f60a',['😊'],'Smiling face with smiling eyes','eoji1f60a'), #5
    ('eoji1f64c','1f64c',['🙌'],'Raising hands','eoji1f64c'),                  #6
    ('eoji1f618','1f618',['😘'],'Face blowing a kiss',None),                   #7
    ('eoji1f384','1f384',['🎄'],'Christmas tree','eoji1f384'),                 #8
    ('eoji1f495','1f495',['💕'],'Two hearts','eoji2764'),                      #9
    ('eoji1f389','1f389',['🎉'],'Party popper','eoji1f389'),                   #10
    ('eoji1f62d','1f62d',['😭'],'Loudly crying face','eoji1f62d'),             #11
    ('eoji1f499','1f499',['💙'],'Blue heart',None),                            #12
    ('eoji2728','2728',['✨'],'Sparkles','eoji2728'),                           #13
    ('eoji2744','2744',['❄️','❄'],'Snowflake','eoji2744'),                     #14
    ('eoji1f60e','1f60e',['😎'],'Face with sunglass','eoji1f60e'),             #15
    ('eoji1f4aa','1f4aa',['💪'],'Flexed biceps','eoji1f4aa'),                  #16
    ('eoji1f64f','1f64f',['🙏'],'Folded hands','eoji1f64f'),                   #17
    ('eoji1f44c','1f44c',['👌'],'OK hands','eoji1f44c'),                       #18
    ('eoji1f48b','1f48b',['💋'],'Kiss mark','eoji1f48b'),                      #19
    ('eoji2600','2600',['☀️','☀'],'Black sun with rays','eoji2600'),           #20
    ('eoji1f4f7','1f4f7',['📷'],'Camera','eoji1f4f7'),                         #21
    ('eoji1f49c','1f49c',['💜'],'Purple heart',None),                          #22
    ('eoji1f609','1f609',['😉'],'Winking face','eoji1f609'),                   #23
    ('eoji1f601','1f601',['😁'],'Beaming face with smiling eyes','eoji1f60a'), #24
    ('eoji1f4f8','1f4f8',['📸'],'Camera with flash','eoji1f4f7'),              #25
    ('eoji1f61c','1f61c',['😜'],'Winking face with tongue','eoji1f61c'),       #26
    # ('eoji1f1fa','1f1fa',['🇺🇸'],'United States','eoji1f1fa'),                #27
]

SKIN_TONES = [
        ('eoji1f3fb', '1f3fb', ['🏻'], 'light skin tone', None),
        ('eoji1f3fc', '1f3fc', ['🏼'], 'medium-light skin tone', None),
        ('eoji1f3fd', '1f3fd', ['🏽'], 'medium skin tone', None),
        ('eoji1f3fe', '1f3fe', ['🏾'], 'medium-dark skin tone', None),
        ('eoji1f3ff', '1f3ff', ['🏿'], 'dark skin tone', None),
]

SKIN_TONES_SET = set(s[2][0] for s in SKIN_TONES)

EOJI_MAPPING = {}
for emoji in EMOJIS:
    EOJI_MAPPING[emoji[0]] = emoji

def eoji_to_name(eoji):
    """Maps from Barbieri emoji representation (e.g., eoji<codepoint>) to name"""
    return EOJI_MAPPING[eoji][3]

def eoji_to_emoji(eoji):
    """Maps from Barbieri emoji representation (e.g., eoji<codepoint>) to emoji unicode character"""
    return EOJI_MAPPING[eoji][2][0]

def merged_eoji(eoji):
    """Returns the actual Barbieri emoji representation (e.g., eoji<codepoint>) that we use.

    Some emojis are merged, this method will return which emoji it is merged to.
    """
    return EOJI_MAPPING[eoji][4]

def to_eoji(emoji):
    return 'eoji{:x}'.format(ord(emoji[0]))

def to_emoji(eoji):
    """Convert eoji format (e.g., eoji2764) into emoji unicode character.
    
    The difference to eoji_to_emoji is that this function is general, while eoji_to_emoji only handles eoji found in
    the emoji list that we use.
    """
    code_point = int(eoji[4:], 16)
    try:
        # For Python 3
        return chr(code_point)
    except:
        # For Python 2
        import struct
        return struct.pack('<I', code_point).decode('utf-32le')

from nltk.tokenize.casual import URLS
url_pattern = re.compile(URLS, re.VERBOSE | re.I | re.UNICODE)
email_pattern = re.compile(r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)')

def is_url(token):
    """Whether the token resembles a URL."""
    return url_pattern.match(token) is not None

def is_email(token):
    """Whether the token resembles an e-mail address."""
    return email_pattern.match(token) is not None

def preprocess_tweet(tweet, remove_mentions=False, remove_urls=False):
    """Preprocess tweet (tokenize, lowercase, remove URLs, remove user mentions)"""
    # Replace @user
    # tweet = re.sub('@[^ ]+', '@user', tweet)
    pattern = re.compile(r"(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){50}(?!@))|(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){1,49})(?![A-Za-z0-9_]*@)")
    if remove_mentions:
        tweet = pattern.sub('', tweet)
    else:
        tweet = pattern.sub(' @user ', tweet)
    # Tokenize
    tokens = casual_tokenize(tweet, preserve_case=False, reduce_len=True, strip_handles=False)
    # Replace URLs
    if remove_urls:
        tokens = [token for token in tokens if not is_url(token)]
        # tokens = [token for token in tokens if not is_email(token)]
    else:
        tokens = ['URL' if is_url(token) else token for token in tokens]
        # tokens = ['EMAIL' if is_email(token) else token for token in tokens]
    result = []
    idx = 0
    while idx < len(tokens):
        if idx+1 < len(tokens) and tokens[idx+1] == 'm' and re.match('\\d+p$', tokens[idx]) is not None:
            result.append('{}{}'.format(tokens[idx], tokens[idx+1]))
            idx += 1
        else:
            result.append(tokens[idx])
        idx += 1
    return ' '.join(result)

ALL_EMOJIS = set(emoji_list.all_emoji)
def strip_emojis(text):
    result = ''
    stripped = []
    for char in text:
        if ((char not in ALL_EMOJIS or re.match('[A-Za-z0-9#*\ufe0f]', char))
                and char != '\u200c'):
            result += char
        else:
            stripped.append(char)
    return result, stripped

def extract_emoji(text):
    """Used in cleaning raw JSON tweet, to extract one emoji from the tweet and deleting others

    If there are multiple emojis, only the last one will be returned.
    """
    the_emoji = None
    for emoji_type, code_point, emoji_list, name, parent in EMOJIS:
        for emoji in emoji_list:
            if emoji in text:
                the_emoji = emoji_type
            text = re.sub(emoji, ' ', text)
    text, stripped = strip_emojis(text)
    text = re.sub('[ \t\r\n]+', ' ', text)
    return text, the_emoji, stripped

def main():
    tweet = 'LoL 😂 @ West Covina, California https://t.co/ylNndaC0ls'
    print('original:', tweet)
    tweet, eoji, stripped = extract_emoji(tweet)
    print('tweet:', tweet)
    print('eoji:', eoji)
    print('stripped:', stripped)
    tweet = preprocess_tweet(tweet)
    print('final:', tweet)

if __name__ == '__main__':
    main()
