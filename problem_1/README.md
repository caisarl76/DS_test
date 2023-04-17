### DS test / Computer Vision and Deep Learning

## Problem 1: Classification on CIFAR-10 dataset

cuda device 쓰기

Augmentation, scheduler 사용하기

model.state_dict 저장하기 
checkpoint 형태 정해주기 
checkpoint = {
    'state_dict': model.state_dict(),
    'epochs':epochs,
    'acc': best_acc,
}

evaluate 에서 해당 state_dict를 읽어와서 load를 하고 eval loop 돌려서 모델 acc 측정,
baseline으로 지정해놓은 성능 및 augmentation+scheduler 성능에 근접한지 확인
