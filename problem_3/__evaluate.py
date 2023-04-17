def evaluate(score):
    passed = True if score > 0.7 else False
    return {'passed': passed, 'score':score}