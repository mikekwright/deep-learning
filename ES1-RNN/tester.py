import json
import re

from collections import Counter
from pprint import pprint


def tweets(file='/data/twitter/emoji_tweets1.json'):
    with open(file, 'r') as infile:
        for line in infile:
            yield json.loads(line)


def tweet_tags():
    return [tag[1:] for t in tweets()
                    for tag in t['text'].split() if len(tag) > 1 and tag[0] == '#']

def tag_counts(debug=False):
    tag_counts = Counter(tweet_tags())
    if debug:
        print(len(tag_counts))
        pprint(tag_counts.most_common(10))
    return tag_counts


def init_test_set(count=60, debug=False):
    initial_set = [[t[0], '0'*len(t[0])] for t in tag_counts().most_common()]
    if debug:
        for i, s in enumerate(initial_set[:count]):
            print(f'#         {s[0]}\ns[{i}][1]="{s[1]}"')
    return initial_set[:count]


def apply_labels_to_test_set():
    s = init_test_set()
    #        DolanTwinsNewVideo
    s[0][1]="000010000100100000"
    #        TeamEXO
    s[1][1]="0001000"
    #        ALDUB25thMonthsary
    s[2][1]="000010001000000000"
    #        ImpeachTrump
    s[3][1]="000000100000"
    #        GainWithXtianDela
    s[4][1]="00010001000010000"
    #        1
    s[5][1]="0"
    #        LOVE_YOURSELF
    s[6][1]="0001100000000"
    #        GameOfThrones
    s[7][1]="0001010000000"
    #        MGWV
    s[8][1]="0000"
    #        MAGA
    s[9][1]="0000"
    #         MTVHottest
    s[10][1]="0010000000"
    #         MSGOnlineGurukul✨🎊
    s[11][1]="001000001000000110"
    #         camilaontour
    s[12][1]="000001010000"
    #         MSGOnlineGurukul,
    s[13][1]="00100000100000010"
    #         AnoSagot
    s[14][1]="00100000"
    #         BTS
    s[15][1]="000"
    #         NationalRelaxationDay
    s[16][1]="000000010000000001000"
    #         havana
    s[17][1]="000000"
    #         omg
    s[18][1]="000"
    #         TeenChoice
    s[19][1]="0001000000"
    #         FOLLOW
    s[20][1]="000000"
    #         UCL
    s[21][1]="000"
    #         Charlottesville
    s[22][1]="000000000000000"
    #         워너원
    s[23][1]="000"
    #         RT
    s[24][1]="00"
    #         LFC
    s[25][1]="000"
    #         TuesdayThoughts
    s[26][1]="000000100000000"
    #         TuesdaySelfie
    s[27][1]="0000001000000"
    #         Trump
    s[28][1]="00000"
    #         방탄소년단
    s[29][1]="00000"
    #         CBB
    s[30][1]="000"
    #         MzanziFolloTrain
    s[31][1]="0000010000100000"
    #         Angel
    s[32][1]="00000"
    #         DolansTwinNewVideo
    s[33][1]="000001000100100000"
    #         2
    s[34][1]="0"
    #         findom
    s[35][1]="000000"
    #         FOLLOWTRICK
    s[36][1]="00000100000"
    #         BackToYou
    s[37][1]="000101000"
    #         OMG
    s[38][1]="000"
    #         MSGOnlineGurukul
    s[39][1]="0010000010000000"
    #         NationalRelaxationDay!
    s[40][1]="0000000100000000010010"
    #         황민현
    s[41][1]="000"
    #         100minutesNonStopOnQ100?
    s[42][1]="001000000100100010100010"
    #         RESIST
    s[43][1]="000000"
    #         SDLive
    s[44][1]="010000"
    #         Antifa
    s[45][1]="000000"
    #         glee
    s[46][1]="0000"
    #         몬스타엑스
    s[47][1]="00000"
    #         WANNAONE
    s[48][1]="00001000"
    #         EXO
    s[49][1]="000"
    #         RETWEET
    s[50][1]="0000000"
    #         TwoGhosts
    s[51][1]="001000000"
    #         ChoiceInternationalArtist
    s[52][1]="0000010000000000001000000"
    #         ATLpajamaJAM
    s[53][1]="001000001000"
    #         NowPlaying
    s[54][1]="0010000000"
    #         Eclipse2017
    s[55][1]="00000010000"
    #         BachelorInParadise
    s[56][1]="000000010100000000"
    #         HAHN
    s[57][1]="0000"
    #         박지훈
    s[58][1]="000"
    #         MzansiFolloTrain
    s[59][1]="0000010000100000"

    updated_set = s
    return [[s[0], [int(c) for c in s[1]]] for s in updated_set]


def tokenizer(string, binary):
    words = []
    current_word = ''
    for i, c in enumerate(string):
        current_word += c
        if binary[i] == 1:
            words.append(current_word)
            current_word = ''

    words.append(current_word)
    return words


TEST_SET=apply_labels_to_test_set()

def validate_model(model, test_set=TEST_SET):
    correct = 0
    for test in test_set:
        expected = test[1]
        actual = model.predict(test[0])
        if actual == expected:
            correct += 1

    return correct / len(test_set)

def model_results(model, test_set=TEST_SET):
    results = []
    for test in test_set:
        output = model.predict(test[0])
        results.append(tokenizer(test[0], output))

    return results


class NeverSplitModel:
    def predict(self, X):
        return [0] * len(X)


class CamelRegexSplit:
    REGEX = '[a-z][A-Z]'
    def predict(self, X):
        results = [0] * len(X)
        for m in re.finditer(CamelRegexSplit.REGEX, X):
            results[m.start()] = 1
        return results


def run_test_models():
    print(validate_model(NeverSplitModel(), TEST_SET))
    model = CamelRegexSplit()
    pprint(model_results(model, TEST_SET))
    print(validate_model(model, TEST_SET))
