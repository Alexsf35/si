import numpy as np


def tanimoto_similarity(x: list[int] , y:list[list[int]]) -> list[float]: 
    """
    Compute the Tanimoto similarity between a binary vector x
    and each binary vector in y.
    """
    similaritys=[]
    sum_x=sum(x)

    for sample in y:
        sum_y=sum(sample)
        intersection=sum([1 for i in range(len(x)) if x[i]== 1 and sample[i]==1])
        similaritys.append(intersection/(sum_x + sum_y - intersection))

    return similaritys
            