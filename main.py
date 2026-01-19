import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from image_helper import *
from parse_xml_annotations import *
from features import *
from reinforcement import *
from metrics import *
from collections import namedtuple
import time
import os
import numpy as np
import random



path_voc = "D:/coding/Multi_object_detection_reinforcement/dataset/VOCdevkit"
# get models 
print("load models")

model_vgg = getVGG_16bn("./models")
model_vgg = model_vgg.cuda()
model = get_q_network()
model = model.cuda()
#2015dqn 적용 model -> current_q_network, target_q_network
current_q_network = model
target_q_network = model

# define optimizers for each model
optimizer = optim.Adam(model.parameters(),lr=1e-6)
criterion = nn.MSELoss().cuda()   

# get image datas
path_voc_1 = "D:/coding/Multi_object_detection_reinforcement/dataset/VOCdevkit/VOC2007"
path_voc_2 = "D:/coding/Multi_object_detection_reinforcement/dataset/VOCdevkit/VOC2012"
class_object = 'aeroplane' #aeroplane(1) dog(12)
image_names_1, images_1 = load_image_data(path_voc_1, class_object)
image_names_2, images_2 = load_image_data(path_voc_2, class_object)
image_names = image_names_1 + image_names_2
images = images_1 + images_2

print("aeroplane_trainval image:%d" % len(image_names))

# define the Pytorch Tensor
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# define the super parameter
multi_agent = 2
epsilon = 1.5
BATCH_SIZE = 100
GAMMA = 0.90
CLASS_OBJECT = 1
steps = 15
epochs = 100
memory = ReplayMemory(1000)

def select_action(state):
    if random.random() < epsilon:
        action = np.random.randint(1,7)
    else:
        qval = model(Variable(state)) #epsilon보다 크다면 모델에 넣고 나오는 액션을 택한다 
        _, predicted = torch.max(qval.data,1)
        action = predicted[0] + 1
    return action


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
def optimizer_model(step, current_q_network, target_q_network):
    if len(memory) < BATCH_SIZE: #배치사이즈 보다 메모리가 작으면 리턴 
        return
    transitions = memory.sample(BATCH_SIZE) #메모리에서 샘플을 뽑아옴 
    batch = Transition(*zip(*transitions)) #뽑아온 샘플을 배치로 다시 바꿔준다
    
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    next_states = [s for s in batch.next_state if s is not None]
    non_final_next_states = Variable(torch.cat(next_states), 
                                     requires_grad=True).type(Tensor)
    state_batch = Variable(torch.cat(batch.state)).type(Tensor)
    action_batch = Variable(torch.LongTensor(batch.action).view(-1,1)).type(LongTensor)
    reward_batch = Variable(torch.FloatTensor(batch.reward).view(-1,1)).type(Tensor)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = current_q_network(state_batch).gather(1, action_batch) #2015dqn 적용 model -> current_q_network
    
    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE, 1).type(Tensor)) 
    next_state_values[non_final_mask.view(-1,1)] = target_q_network(non_final_next_states).max(1)[0]  #2015dqn 적용 model -> target_q_network
    
    
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
  
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values.detach() * GAMMA) + reward_batch
    
    # Compute  loss
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #일정 간격으로 타겟 q네트워크 업데이트
    if step%5==0:
        target_q_network.load_state_dict(current_q_network.state_dict())

# train procedure
print('train the Q-network')
for epoch in range(epochs): #에폭만큼 반복 
    print('epoch: %d' %epoch)
    now = time.time()
    for i in range(len(image_names)): #전체 이미지 반복 
        # the image part
        image_name = image_names[i]
        image = images[i]
        if i < len(image_names_1):
            annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc_1) #xml파일에서 gt box 좌표들 가져온다
        else:
            annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc_2)           
        classes_gt_objects = get_ids_objects_from_annotation(annotation)
        gt_masks = generate_bounding_box_from_annotation(annotation, image.shape) #gt들을 다 가져온다
         
        # the iou part 코드 정리 필요
        original_shape = (image.shape[0], image.shape[1]) #이미지의 원래 모양을 저장 높이와 너비를 담은 튜플
        #region_mask = np.ones((image.shape[0], image.shape[1])) #이미지와 동일한 크기를 가진 모든 요소가 1인 배열
        #choose the max bouding box    
        #iou = find_max_bounding_box(gt_masks, region_mask, classes_gt_objects, CLASS_OBJECT) #gt마스크들과 예측 마스크들을 비교하여 가장 iou가 높은 iou추출
        
        region_image =[]
        size_mask=[]
        offset=[] 
        history_vector=[] 
        region_mask=[] 
        state=[]
        done=[] 
        init_iou=[]
       
        # the initial part
        for agent in range(multi_agent):
            region_image.append(image) #처음 이미지 자체
            size_mask.append(original_shape) #이미지 shape
            offset.append((0, 0))
            history_vector.append(torch.zeros((4,6)))
            region_mask.append(np.ones((image.shape[0], image.shape[1])))
            state.append(get_state(region_image[agent], history_vector[agent], model_vgg))
            done.append(False)
            init_iou.append(0)
        
            
        for step in range(steps):
            
            reward_iou_list = []
            reward_list = []
            agent_gt = []
            
            for agent in range(multi_agent):
                reward_list.append(0) # 각 스텝마다 모든 agent들의 리워드들 합을 위해 빈 리스트 생성
                reward_iou_list.append(0)
                agent_gt.append(0)
                
            for agent in range(multi_agent):
                '''
                if step == 0: #첫번째 스텝에서는 액션을 랜덤으로 실행 이렇게 해주어야 박스들간의 거리를 다양하게 두어 여러가지 gt를 찾을 수 있다. 굳이 할 필요 있는지 염두 필요
                    action == np.random.randint(1,7)
                else:            
                '''
                if state[agent] == None:
                    continue
                
                '''
                #가장 가까운 distance를 가지고 목표 box잡기
                closet_gt = closet_distance_gt(gt_masks, region_mask[agent], classes_gt_objects, CLASS_OBJECT, annotation, size_mask[agent]) #현재 box와 가장 가까운 gtbox를 찾아옴 
                reward_iou = calculate_iou(region_mask[agent], closet_gt) #그 gt box와 현재 box의 iou계산
                reward_iou_list[agent] = reward_iou
                print(reward_iou_list)
                '''
                
                #max iou를 이용하여 목표 box 잡기 (현재 찾을려는 gt를 찾기 위하여 find함수에서 gt mask를 가져오게 코드 수정)
                reward_iou, agent_gt[agent] = find_max_bounding_box_and_gt(gt_masks, region_mask[agent], classes_gt_objects, CLASS_OBJECT, annotation)
                reward_iou_list[agent]=reward_iou
                print(reward_iou_list)
                print(agent_gt)
                
                #action은 취하지 않고 일단 reward부터 계산
                if reward_iou > 0.5:
                   
                    #찾은 객체가 다를 경우 reward를 더 크게 주기
                        
                    agent_reward = get_reward_trigger(reward_iou) 
                    reward_list[agent] = agent_reward
                else:
                    agent_reward = get_reward_movement(init_iou[agent], reward_iou) #새로 계산한 iou가 더 높으면 reward +1 아니면 reward-1
                    reward_list[agent] = agent_reward #reward list에 추가   
                    init_iou[agent] = reward_iou                 
            multi_reward = get_multi_reward_sum(reward_list) #다른 에이전트들의 reward 총합 
            
            '''
            #gtmask가 다를 경우 reward +3 (현재 agent가 2개일 때는 문제 없지만 3개이상이 되면 agent가 객체를 찾으면 원소가 모두 0이 되므로 reward가 계속 -1씩 깎인다.)
            if len(agent_gt)==len(set(agent_gt)):
                multi_reward = multi_reward+1
            else:
                multi_reward = multi_reward-1 #위의 문제를 임시로 해결하고자 -1 제거
            '''   
              
                
            #agent들끼리 IoU가 최대한 다르게끔 reward
            agent_iou = calculate_iou(region_mask[agent-1], region_mask[agent-2]) #region mask는 agent가 종료된 시점에서 그 mask가 다음 step도 유지 된다
            if agent_iou > 0.8:
                multi_reward = multi_reward-1
            else:
                multi_reward = multi_reward+1    
            
            

            
            # 다시 agent 반복문 시작
            for agent in range(multi_agent):
                if state[agent] == None:
                    continue
                                
                '''
                #distance로 gt 박스잡는 알고리즘
                closet_gt = closet_distance_gt(gt_masks, region_mask[agent], classes_gt_objects, CLASS_OBJECT, annotation, size_mask[agent]) #현재 box와 가장 가까운 gtbox를 찾아옴 
                iou = calculate_iou(region_mask[agent], closet_gt) #그 gt box와 현재 box의 iou계산
                '''
                
                # Select action, the author force terminal action if case actual IoU is higher than 0.5
                if reward_iou_list[agent] > 0.5:
                    action = 6
                else:
                    action = select_action(state[agent])
                
                # Perform the action and observe new state
                if action == 6:
                    next_state = None #next state도 agent 반복문에서 생성 되어 나중에 state로 들어가기 때문에 굳이 list indext에 저장해줄 필요 x
                    done[agent] = True
                else:
                    offset[agent], region_image[agent], size_mask[agent], region_mask[agent] = get_crop_image_and_mask(original_shape, offset[agent],  #action과 현재 이미지를 가지고 크롭한 이미지와 마스크 생성
                                                                    region_image[agent], size_mask[agent], action) #action은 이전 action내용을 가져 오지 않고 반복문 안에서 구하므로 굳이 인데스 저장 필요 x
                    # update history vector and get next state
                    history_vector[agent] = update_history_vector(history_vector[agent], action) #이번에 행한 action을 history vector에 추가
                    next_state = get_state(region_image[agent], history_vector[agent], model_vgg) # 크롭이미지와 history를 이용하여 state 가져옴
                    
                    # find the max bounding box in the region image
                    #closet_gt = closet_distance_gt(region_mask, gt_masks) #현재 box와 가장 가까운 gtbox를 찾아옴
                    #new_iou = calculate_iou(region_mask, closet_gt) #그 gt box와 현재 box의 iou계산 
                    
                    #new_iou = find_max_bounding_box(gt_masks, region_mask, classes_gt_objects, CLASS_OBJECT)
                    #reward = get_reward_movement(iou, new_iou) #새로 계산한 iou가 더 높으면 reward +1 아니면 reward-1
                    #reward_list.append(reward) #reward list에 추가
                    #iou = new_iou
                reward = multi_reward #reward가 멀티에이전트 다 받아들이도록 수정    
                    
                # Store the transition in memory
                memory.push(state[agent], action-1, next_state, reward)
                
                # Move to the next state
                state[agent] = next_state
                
                # Perform one step of the optimization (on the target network)
                # 2015 DQN 방식 적용 5step 마다 target_q 업데이트
                optimizer_model(step, current_q_network, target_q_network) #각 agent마다 학습 하는 방식 >> step종료시에 학습하는 방식  
            print('epoch: %d, image: %d, step: %d, reward: %d' %(epoch ,i, step, reward))
            for status in done: #모든 done이 true이면 step 반복문 종료
                if not status:
                    break
            else:
                break    
    if epsilon > 0.1:
        epsilon -= 0.1
    time_cost = time.time() - now
    print('epoch = %d, time_cost = %.4f' %(epoch, time_cost))
    
# save the whole model
Q_NETWORK_PATH = './models/' + 'voc2012_2007_model_aeroplane_epsilon1.5_epoch100_iou_2agent_detach_far'
torch.save(current_q_network, Q_NETWORK_PATH) # model -> current_q_network 로 변경 
print('Complete')