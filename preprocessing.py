"""
In this module we try to use an LSTM(Long Short Term Memory) network for language modelling in
a similar way that we used the previous vanilla rnn. Hopefully we will see better results.
"""
from lstm_tf import LSTM
# Prepare data.
# data path
PAUL_GRAHAM_DATA = 'data/paulg/paulg.txt'
REDDIT_COMMENTS_DATA = 'data/RedditComments/reddit-comments-2015-08.txt'
TESTING_DATA = 'data/small_data_for_testing.txt'
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
LINES = DATA.split('\n')
BATCH_SIZE = 256 if len(DATA) > 256 else len(DATA)
MODEL = LSTM(20, 30, batch_size=BATCH_SIZE)
MODEL.train(DATA, CHARACTER_TO_INDEX, number_epox=10)
