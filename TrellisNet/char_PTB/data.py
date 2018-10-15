import observations
import os
import pickle
from utils import Corpus

def data_generator(args):
    file, testfile, valfile = getattr(observations, args.dataset)('data/')
    file, testfile, valfile = file.replace('<eos>', chr(255)), testfile.replace('<eos>', chr(255)), valfile.replace(
        '<eos>', chr(255))  # Just replace <eos> with another unusual alphabet here (that is not in PTB)
    file_len = len(file)
    valfile_len = len(valfile)
    testfile_len = len(testfile)

    ############################################################
    # Use the following if you want to pickle the loaded data

    pickle_name = "{0}.corpus".format(args.dataset)
    if os.path.exists(pickle_name):
        print("Loading cached data...")
        corpus = pickle.load(open(pickle_name, 'rb'))
    else:
        corpus = Corpus(file + " " + valfile + " " + testfile)
        pickle.dump(corpus, open(pickle_name, 'wb'))
    ############################################################

    return file, file_len, valfile, valfile_len, testfile, testfile_len, corpus