from scipy.stats import entropy

print(entropy([5/14,9/14],base=2))
print(entropy([2/9,7/9],base=2))
print(entropy([3/5,2/5],base=2))
gain_info = entropy([5/14,9/14],base=2)-9/14*entropy([2/9,7/9],base=2)-5/14*entropy([3/5,2/5],base=2)
print(gain_info)