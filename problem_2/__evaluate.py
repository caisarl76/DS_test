import numpy as np
import time 

from __solution import solution as fast_solution
from __template import solution

def evaluate():
    X = np.random.randint(255, size=(224, 224))
    filters = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    
    start = time.time()
    
    sol_out = fast_solution(X, filters)
    sol_time = time.time() - start
    
    start = time.time()
    user_out = solution(X, filters)
    user_time = time.time() - start
    
    time_ratio = user_time / sol_time
    func_answer = True if np.sum((sol_out == user_out).astype(int)) > (224*224*0.98) else False
    
    if func_answer and time_ratio < 1.2:
        return {'passed': True, 'score': time_ratio}
    else:
        return {'passed': False, 'score': time_ratio}
    
if __name__=='__main__':
    evaluate()