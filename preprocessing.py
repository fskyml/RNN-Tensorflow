"""
In this module we try to use an LSTM(Long Short Term Memory) network for language modelling in
a similar way that we used the previous vanilla rnn. Hopefully we will see better results.
"""

# Prepare data.
# data path
PAUL_GRAHAM_DATA = '/home/arko/Documents/Datasets/paulg/paulg.txt'
REDDIT_COMMENTS_DATA = '/home/arko/Documents/Datasets/RedditComments/reddit-comments-2015-08.txt'
DATA = open(
    PAUL_GRAHAM_DATA, 'r', encoding='utf8'
).read()

# Extract individuals characters from the string.
CHARACTERS = sorted(
    list(
        set(DATA)
    )
)

# Prepare a mapping from characters to index
CHARACTER_TO_INDEX = {
    ch: i for i, ch in enumerate(CHARACTERS)
}

# and vice versa
INDEX_TO_CHARACTER = {
    i: ch for i, ch in enumerate(CHARACTERS)
}
