import pandas as pd
from .domain import Dataset, Biclusters
from .mining import BicPAMS, Miner
from .mapping import Mapper
from .closing import Closer

#normcols, frequency

def constant_patterns():
    data = Dataset(data='data/matrix.csv', transpose=False)
    mapper = Mapper(strategy = 'flexible', bins = 5, noise=True, removals=[]) #normcols
    miner = Miner(pattern='constant', stopConditions={'mincols': 4, 'minbics': 10, 'minsig':1}, niterations=2)
    closer = Closer(mergeOverlap=0.7, filterOverlap=0.7, order='area')
    bics = BicPAMS.run(data, mapper, miner, closer)
    print(Biclusters.to_string(bics, data, detail=False))
    open('output_constant.txt', 'a').write(Biclusters.to_string(bics, data, detail=True, disc=mapper.discata))

def spm_patterns():
    data = Dataset(data='data/gyeast.arff', transpose=False)
    mapper = Mapper(strategy = 'normrows', bins = 20)
    miner = Miner(pattern='orderpreserving', stopConditions={'mincols': 4, 'minbics': 10, 'minsig':1}, niterations=2)
    closer = Closer(mergeOverlap=0.8, filterOverlap=0.7, order='area')  #) #,
    bics = BicPAMS.run(data, mapper, miner, closer)
    print(Biclusters.to_string(bics, data, detail=False))
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', None)
    open('output_order.txt', 'a').write(Biclusters.to_string(bics, data, detail=True))

if __name__ == '__main__':
    constant_patterns()
    spm_patterns()