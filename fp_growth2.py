"""
    This code is copied from Andrew9255,
    https://gist.github.com/Andrew9255/7d75710d973e64a33187fca08c273840
"""
import pandas as pd
import os


class node:
    def __init__(self, word, word_count=0, parent=None, link=None):
        self.word = word
        self.word_count = word_count
        self.parent = parent
        self.link = link
        self.children = {}

    # # tree traversal
    # def visittree(self):
    #     #        if self is None:
    #     #            return None
    #     output = []
    #     output.append(str(vocabdic[self.word]) + " " + str(self.word_count))
    #     if len(list(self.children.keys())) > 0:
    #         for i in (list(self.children.keys())):
    #             output.append(self.children[i].visittree())
    #     return output


'''      Build FPTREE class and method       '''


class fptree:
    def __init__(self, data, minsup=400):
        # raw data and minminual support
        self.data = data
        self.minsup = minsup

        # null root
        self.root = node(word="Null", word_count=1)

        # each line of transaction with new order from the most frequent items to less
        self.wordlinesort = []
        # node table containing link of all nodes of same word
        self.nodetable = []
        # dictionary contaiing word more than the minsupport count with des order
        self.wordsortdic = []

        # dictionaly containing word and the support count
        self.worddic = {}
        # dictionary with word and it's postion of the support count rank
        self.wordorderdic = {}
        #
        #        self.preprocess(data)
        #        #first scan to build all the necessay dictionary
        self.construct(data)
        # second scan and build fp tree line  by line

    def construct(self, data):
        # get support count for all word
        for tran in data:
            for words in tran:
                if words in self.worddic.keys():
                    self.worddic[words] += 1
                else:
                    self.worddic[words] = 1
        wordlist = list(self.worddic.keys())
        # prune all the world with < min support count
        for word in wordlist:
            if (self.worddic[word] < self.minsup):
                del self.worddic[word]
        # sort the remaing items des, with first word count than work#id
        self.wordsortdic = sorted(self.worddic.items(), key=lambda x: (-x[1], x[0]))
        # create a table containing word, wordcount and all link node of that word
        t = 0
        for i in self.wordsortdic:
            word = i[0]
            wordc = i[1]
            self.wordorderdic[word] = t
            t += 1
            wordinfo = {'wordn': word, 'wordcc': wordc, 'linknode': None}
            self.nodetable.append(wordinfo)
        # construct fptree line by line

        for line in data:
            supword = []
            for word in line:
                # only keep words with support count higher than minsupport
                if word in self.worddic.keys():
                    supword.append(word)
            # insert words to the fp tree
            if len(supword) > 0:
                # reorder the words
                sortsupword = sorted(supword, key=lambda k: self.wordorderdic[k])
                self.wordlinesort.append(sortsupword)
                # enter the word one by one from begining
                R = self.root
                #                print(sortsupword)
                for i in sortsupword:
                    if i in R.children.keys():
                        R.children[i].word_count += 1
                        R = R.children[i]
                    else:

                        R.children[i] = node(word=i, word_count=1, parent=R, link=None)
                        R = R.children[i]
                        # link this node to nodetable
                        for wordinfo in self.nodetable:
                            if wordinfo["wordn"] == R.word:
                                # find the last node of the  node linklist
                                if wordinfo["linknode"] is None:
                                    wordinfo["linknode"] = R
                                else:
                                    iter_node = wordinfo["linknode"]
                                    while iter_node.link is not None:
                                        iter_node = iter_node.link
                                    iter_node.link = R

    # create transactions for conditinal tree
    def condtreetran(self, N):
        if N.parent is None:
            return None

        condtreeline = []
        # starting from the leaf node reverse add word till hit root
        while N is not None:
            line = []
            PN = N.parent
            while PN.parent is not None:
                line.append(PN.word)
                PN = PN.parent
            # reverse order the transaction
            line = line[::-1]
            for i in range(N.word_count):
                condtreeline.append(line)
                # move on to next linknode
            N = N.link
        return condtreeline

    # Find frequent word list by creating conditional tree
    def findfqt(self, parentnode=None):
        if len(list(self.root.children.keys())) == 0:
            return None
        result = []
        sup = self.minsup
        # starting from the end of nodetable
        revtable = self.nodetable[::-1]
        for n in revtable:
            fqset = [set(), 0]
            if (parentnode == None):
                fqset[0] = {n['wordn'], }
            else:
                fqset[0] = {n['wordn']}.union(parentnode[0])
            fqset[1] = n['wordcc']
            result.append(fqset)
            condtran = self.condtreetran(n['linknode'])
            # recursively build the conditinal fp tree
            contree = fptree(condtran, sup)
            conwords = contree.findfqt(fqset)
            if conwords is not None:
                for words in conwords:
                    result.append(words)
        return result

    # check if tree hight is larger than 1
    def checkheight(self):
        if len(list(self.root.children.keys())) == 0:
            return False
        else:
            return True


def test(data_or_file=None, support_count=4):
    """

    :param data_or_file:
    :param support_count:
    :return: a pandas DataFrame
    """
    if data_or_file is None:
        data = [['I1', 'I2', 'I5'],
                ['I2', 'I4'],
                ['I2', 'I3'],
                ['I1', 'I2', 'I4'],
                ['I1', 'I3'],
                ['I2', 'I3'],
                ['I1', 'I3'],
                ['I1', 'I2', 'I3', 'I5'],
                ['I1', 'I2', 'I3']]
        message = "test"
    elif os.path.exists(data_or_file):
        # this declaim data is a path of csv file
        data_origin = pd.read_csv(data_or_file)["items"].values
        data = []
        for i in data_origin:
            data.append(i.split(','))
        message = "data"
    else:
        data = data_or_file
        message = "data"

    fp_tree = fptree(data, support_count)  # create FP tree on data

    # print(f"\n========== Printing Frequent Word Set on {message} ==========")
    frequentwordset = fp_tree.findfqt()  # mining frequent patt
    frequentwordset = sorted(frequentwordset, key=lambda k: -k[1])
    d = {"items": [], "support_count": [], "set_size": []}
    for item in frequentwordset:
        d["items"].append(list(item[0])[0]) if len(item[0]) == 1 else d["items"].append(tuple(item[0]))
        d["support_count"].append(item[1])
        d["set_size"].append(len(item[0]))
    return pd.DataFrame(data=d)


def main():
    test()


# # print frequent patt
# for word in frequentwordset:
#     count = (str(word[1]) + "\t")
#     words = ''
#     for val in word[0]:
#         words += (str(vocabdic[val]) + " ")
#     print(count + words)
#
# # print conditional fp tree height >1
# for i in fp_tree.nodetable[::-1]:
#     lines = fp_tree.condtreetran(i['linknode'])
#     condtree = fptree(lines, min_sup)
#     if (condtree.checkheight()):
#         print('Condtional FPTree Root on ' + (vocabdic[i['wordn']]))
#         print(condtree.root.visittree())


if __name__ == '__main__':
    main()
