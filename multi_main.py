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
#sigmoid_model = get_multi_q_network()
#sigmoid_model = sigmoid_model.cuda()
#2015dqn 적용 model -> current_q_network, target_q_network
agent1_actual = model
agent1_virtual = model
agent2_actual = model
agent2_virtual = model

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
epochs = 80
memory = ReplayMemory(1000)

def select_action(state): #simoid output 받을수 있게 설정
    if random.random() < epsilon:
        action = np.random.randint(1,7)
    else:#epsilon보다 크다면 모델에 넣고 나오는 액션을 택한다
        qval_agent1_actual = agent1_actual(Variable(state))
        sigmoid_agent1_actual = torch.sigmoid(qval_agent1_actual)
        device = sigmoid_agent1_actual.device
        sigmoid_agent2_virtual = torch.ones(6, device=device) - sigmoid_agent1_actual
        qval_agent1_actual = qval_agent1_actual*sigmoid_agent1_actual
        qval_agent2_virtual = agent2_virtual(Variable(state))*sigmoid_agent2_virtual
        qval = qval_agent1_actual + qval_agent2_virtual
        _, predicted = torch.max(qval.data,1)
        action = predicted[0] + 1
    return action


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'agent1_reward', 'agent2_reward'))
def optimizer_model(agent_model):
    if len(memory) < BATCH_SIZE: #배치사이즈 보다 메모리가 작으면 리턴 
        return
    
    if agent_model == 'agent1_update':
        transitions = memory.sample(BATCH_SIZE) #메모리에서 샘플을 뽑아옴 
        batch = Transition(*zip(*transitions)) #뽑아온 샘플을 배치로 다시 바꿔준다

        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states = Variable(torch.cat(next_states), 
                                            requires_grad=True).type(Tensor)
        state_batch = Variable(torch.cat(batch.state)).type(Tensor)
        action_batch = Variable(torch.LongTensor(batch.action).view(-1,1)).type(LongTensor)
        reward_batch = Variable(torch.FloatTensor(batch.agent1_reward).view(-1,1)).type(Tensor)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = agent1_actual(state_batch).gather(1, action_batch) #2015dqn 적용 model -> current_q_network

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(BATCH_SIZE, 1).type(Tensor)) 
        next_state_values[non_final_mask.view(-1,1)] = agent1_actual(non_final_next_states).max(1)[0]  #2015dqn 적용 model -> target_q_network


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
    
    elif agent_model == 'agent2_update': #수정필요
        transitions = memory.sample(BATCH_SIZE) #메모리에서 샘플을 뽑아옴 
        batch = Transition(*zip(*transitions)) #뽑아온 샘플을 배치로 다시 바꿔준다

        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states = Variable(torch.cat(next_states), 
                                            requires_grad=True).type(Tensor)
        state_batch = Variable(torch.cat(batch.state)).type(Tensor)
        action_batch = Variable(torch.LongTensor(batch.action).view(-1,1)).type(LongTensor)
        reward_batch = Variable(torch.FloatTensor(batch.agent2_reward).view(-1,1)).type(Tensor)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = agent2_virtual(state_batch).gather(1, action_batch) #2015dqn 적용 model -> current_q_network

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(BATCH_SIZE, 1).type(Tensor)) 
        next_state_values[non_final_mask.view(-1,1)] = agent2_virtual(non_final_next_states).max(1)[0]  #2015dqn 적용 model -> target_q_network


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

# train procedure
print('train the Q-network')
for epoch in range(epochs):
    print('epoch: %d' %epoch)
    now = time.time()
    for i in range(len(image_names)):
        # the image part
        image_name = image_names[i]
        image = images[i]
        if i < len(image_names_1):
            annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc_1)
        else:
            annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc_2)            
        classes_gt_objects = get_ids_objects_from_annotation(annotation)
        gt_masks = generate_bounding_box_from_annotation(annotation, image.shape) 
         
        # the iou part
        original_shape = (image.shape[0], image.shape[1])
        region_mask = np.ones((image.shape[0], image.shape[1]))
        #choose the max bouding box
        iou_list = find_iou_list(gt_masks, region_mask, classes_gt_objects, CLASS_OBJECT)
        if len(iou_list) == 1:
            agent1_iou = iou_list[0] #출력으로 첫번째 주석에 대한 iou 
            agent2_iou = iou_list[0]
        else: 
            agent1_iou = iou_list[0]
            agent2_iou = iou_list[1]
        
        # the initial part
        region_image = image
        size_mask = original_shape
        offset = (0, 0)
        history_vector = torch.zeros((4,6))
        state = get_state(region_image, history_vector, model_vgg)
        done = False
        for step in range(steps):
            
            #action 선택
            if agent1_iou > 0.5: #agent1의 iou가 0.5보다 크면 선택 action
                agent1_action = 6
            else:
                agent1_action = select_action(state) #sigmoid 수정필요
            
            # Perform the action and observe new state
            if agent1_action == 6:
                next_state = None 
                done = True
                agent1_reward = get_reward_trigger(agent1_iou)
                agent2_reward = get_reward_trigger(agent2_iou)
                
            else:
                offset, region_image, size_mask, region_mask = get_crop_image_and_mask(original_shape, offset,  #action과 현재 이미지를 가지고 크롭한 이미지와 마스크 생성
                                                                region_image, size_mask, agent1_action) 
                # update history vector and get next state
                history_vector = update_history_vector(history_vector, agent1_action) #이번에 행한 action을 history vector에 추가
                next_state = get_state(region_image, history_vector, model_vgg) # 크롭이미지와 history를 이용하여 state 가져옴
               
                # find the max bounding box in the region image
                new_agent_iou_list = find_iou_list(gt_masks, region_mask, classes_gt_objects, CLASS_OBJECT) # list를 다 가져온다
                
                if len(new_agent_iou_list) == 1:
                    agent1_new_iou = new_agent_iou_list[0] #출력으로 첫번째 주석에 대한 iou 
                    agent2_new_iou = new_agent_iou_list[0]
                else: 
                    agent1_new_iou = new_agent_iou_list[0]
                    agent2_new_iou = new_agent_iou_list[1]
                
                agent1_reward = get_reward_movement(agent1_iou, agent1_new_iou)
                agent2_reward = get_reward_movement(agent2_iou, agent2_new_iou)
               
                
                agent1_iou = agent1_new_iou         
                agent2_iou = agent2_new_iou       
                
            print('epoch: %d, image: %d, step: %d, reward: %d' %(epoch ,i, step, agent1_reward))                 
            
            # Store the transition in memory 메모리 저장 코드 수정
            memory.push(state, agent1_action-1, next_state, agent1_reward, agent2_reward)
            
            #메모리에서 agent1 배치 뽑아내고 업데이트
            optimizer_model('agent1_update') 
            
            #웨이트 가상 에이전트로 복사
            agent1_virtual.load_state_dict(agent1_actual.state_dict()) 
                        
            
            #다른 agent들 학습
            for agnet in range(multi_agent-1):
                
                #메모리에서 agent 2 배치 뽑아내고 업데이트 
                optimizer_model('agent2_update')
                
                #실제 에이전트 웨이트 가상 에이전트로 복사
                agent2_actual.load_state_dict(agent2_virtual.state_dict())
                

            
            # Move to the next state
            state = next_state
            
            # Perform one step of the optimization (on the target network)
            if done:
                break
            
    if epsilon > 0.1:
        epsilon -= 0.1
    time_cost = time.time() - now
    print('epoch = %d, time_cost = %.4f' %(epoch, time_cost))
    
# save the whole model
Q_NETWORK_PATH1 = './models/collabo/' + 'voc_aeroplane_sigmoid_epoch80_agent1_actual'
Q_NETWORK_PATH2 = './models/collabo/' + 'voc_aeroplane_sigmoid_epoch80_agent2_actual'
Q_NETWORK_PATH3 = './models/collabo/' + 'voc_aeroplane_sigmoid_epoch80_agent1_virtual'
Q_NETWORK_PATH4 = './models/collabo/' + 'voc_aeroplane_sigmoid_epoch80_agent2_virtual'

torch.save(agent1_actual, Q_NETWORK_PATH1 )
torch.save(agent2_actual, Q_NETWORK_PATH2 )
torch.save(agent1_virtual, Q_NETWORK_PATH3 )
torch.save(agent2_virtual, Q_NETWORK_PATH4 )
print('Complete')
