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



path_voc = "C:/startcoding/Active-Object-Localization-Deep-Reinforcement-Learning-master/Active-Object-Localization-Deep-Reinforcement-Learning-master/datasets/VOCdevkit/VOC2007"

# get models 
print("load models")

model_vgg = getVGG_16bn("../models")
model_vgg = model_vgg.cuda()
model = get_q_network()
model = model.cuda()

# define optimizers for each model
optimizer = optim.Adam(model.parameters(),lr=1e-6)
criterion = nn.MSELoss().cuda()   

# get image datas
path_voc_1 = "C:/startcoding/Active-Object-Localization-Deep-Reinforcement-Learning-master/Active-Object-Localization-Deep-Reinforcement-Learning-master/datasets/VOCdevkit/VOC2007"
path_voc_2 = "C:/startcoding/Active-Object-Localization-Deep-Reinforcement-Learning-master/Active-Object-Localization-Deep-Reinforcement-Learning-master/datasets/VOCdevkit/VOC2012"
class_object = 'dog' #aeroplane(1) dog(12)
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
epsilon = 1.0
BATCH_SIZE = 100
GAMMA = 0.90
CLASS_OBJECT = 1
steps = 14
change = 4
epochs = 50
alpha = 0.2
nu=3.0
threshold=0.5
steps_done = 0
memory = ReplayMemory(1000)
actions_history = []

def intersection_over_union(box1, box2):
        """
            Calcul de la mesure d'intersection/union IOU계산
            Entrée :
                Coordonnées [x_min, x_max, y_min, y_max] de la boite englobante de la vérité terrain et de la prédiction (gt와 예측 상자의 좌표)
            Sortie :
                Score d'intersection/union.

        """
        x11, x21, y11, y21 = box1
        x12, x22, y12, y22 = box2
        
        yi1 = max(y11, y12)
        xi1 = max(x11, x12)
        yi2 = min(y21, y22)
        xi2 = min(x21, x22)
        inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x22 - x12) * (y22 - y12)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

def select_h_action(state):
    if random.random() < epsilon:
        action = np.random.randint(1,7)
    else:
        qval = model(Variable(state))
        _, predicted = torch.max(qval.data,1)
        action = predicted[0] + 1
    return action


def select_d_action(state, actions, ground_truth):
        """
            Selection de l'action dépendemment de l'état
            상태에 따른 동작 선택
            Entrée :
                - Etat actuel. 
                - Vérité terrain.
            Sortie :
                - Soi l'action qu'aura choisi le modèle soi la meilleure action possible ( Le choix entre les deux se fait selon un jet aléatoire ).
        """
        sample = random.random()
        eps_threshold = epsilon
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                if use_cuda:
                    inpu = Variable(state).cuda()
                else:
                    inpu = Variable(state)
                qval = model(inpu)
                _, predicted = torch.max(qval.data,1)
                action = predicted[0] + 1 # + 1
                try:
                  return action.cpu().numpy()[0]
                except:
                  return action.cpu().numpy()
        else:
            #return np.random.randint(0,9)   # Avant implémentation d'agent expert
            return get_best_next_action(actions, ground_truth) # Appel à l'agent expert.

def get_best_next_action(actions, ground_truth):
        """
            Implémentation de l'Agent expert qui selon l'état actuel et la vérité terrain va donner la meilleur action possible.
            현재 상태와 실제 상황에 따라 가능한 최선의 조치를 제공하는 Expert Agent 구현.
            Entrée :
                - Liste d'actions executées jusqu'à présent.
                - Vérité terrain.
            Sortie :
                - Indice de la meilleure action possible.

        """
        max_reward = -99
        best_action = -99
        positive_actions = []
        negative_actions = []
        actual_equivalent_coord = calculate_position_box(actions)
        for i in range(0, 9):
            copy_actions = actions.copy()
            copy_actions.append(i)
            new_equivalent_coord = calculate_position_box(copy_actions)
            if i!=0:
                reward = compute_reward(new_equivalent_coord, actual_equivalent_coord, ground_truth)
            else:
                reward = compute_trigger_reward(new_equivalent_coord,  ground_truth)
            
            if reward>=0:
                positive_actions.append(i)
            else:
                negative_actions.append(i)
        if len(positive_actions)==0:
            return random.choice(negative_actions)
        return random.choice(positive_actions)

def calculate_position_box(actions, xmin=0, xmax=224, ymin=0, ymax=224):
        """
            Prends l'ensemble des actions depuis le début et en génére les coordonnées finales de la boite englobante.
            처음부터 모든 작업을 수행하고 경계 상자의 최종 좌표를 생성합니다.
            Entrée :
                - Ensemble des actions sélectionnées depuis le début.
            Sortie :
                - Coordonnées finales de la boite englobante.
        """
        # Calcul des alpha_h et alpha_w mentionnées dans le papier
        alpha_h = alpha * (  ymax - ymin )
        alpha_w = alpha * (  xmax - xmin )
        real_x_min, real_x_max, real_y_min, real_y_max = 0, 224, 0, 224

        # Boucle sur l'ensemble des actions
        for r in actions:
            if r == 1: # Right
                real_x_min += alpha_w
                real_x_max += alpha_w
            if r == 2: # Left
                real_x_min -= alpha_w
                real_x_max -= alpha_w
            if r == 3: # Up 
                real_y_min -= alpha_h
                real_y_max -= alpha_h
            if r == 4: # Down
                real_y_min += alpha_h
                real_y_max += alpha_h
            if r == 5: # Bigger
                real_y_min -= alpha_h
                real_y_max += alpha_h
                real_x_min -= alpha_w
                real_x_max += alpha_w
            if r == 6: # Smaller
                real_y_min += alpha_h
                real_y_max -= alpha_h
                real_x_min += alpha_w
                real_x_max -= alpha_w
            if r == 7: # Fatter
                real_y_min += alpha_h
                real_y_max -= alpha_h
            if r == 8: # Taller
                real_x_min += alpha_w
                real_x_max -= alpha_w
        real_x_min, real_x_max, real_y_min, real_y_max = rewrap(real_x_min), rewrap(real_x_max), rewrap(real_y_min), rewrap(real_y_max)
        return [real_x_min, real_x_max, real_y_min, real_y_max]

def rewrap(coord):
        return min(max(coord,0), 224)

def compute_trigger_reward(actual_state, ground_truth):
        """
            Calcul de la récompensée associée à un état final selon les cas. 경우에 따라 최종상태와 관련된 보상 계산
            Entrée :
                Etat actuel et boite englobante de la vérité terrain
            Sortie : 
                Récompense attribuée
        """
        res = intersection_over_union(actual_state, ground_truth)
        if res>=threshold:
            return nu
        return -1*nu
    
def compute_reward(actual_state, previous_state, ground_truth):
        """
            Calcul la récompense à attribuer pour les états non-finaux selon les cas. 경우에 따라 최종이 아닌 상태에 대해 수여되는 보상
            Entrée :
                Etats actuels et précédents ( coordonnées de boite englobante )
                Coordonnées de la vérité terrain
            Sortie :
                Récompense attribuée
        """
        res = intersection_over_union(actual_state, ground_truth) - intersection_over_union(previous_state, ground_truth)
        if res <= 0:
            return -1
        return 1    
    
def get_max_bdbox(ground_truth_boxes, actual_coordinates ):
        """
            Récupére parmis les boites englobantes vérité terrain d'une image celle qui est la plus proche de notre état actuel.
            현재 상태에 가장 가까운 이미지의 실측 경계 상자에서 검색합니다.
            Entrée :
                - Boites englobantes des vérités terrain.
                - Coordonnées actuelles de la boite englobante.
            Sortie :
                - Vérité terrain la plus proche.
        """
        max_iou = False #최대 IoU값을 나타내는 변수를 False로 초기화
        max_gt = [] #최대 IoU를 가진 정답 상자를 저장할 변수를 빈 리스트로 초기화
        for gt in ground_truth_boxes: #주어진 정답 상자들에 대해 반복
            iou = intersection_over_union(actual_coordinates, gt) #현재 상태와 정답 상자사이의 교차하는 부분의 비율을 계산
            if max_iou == False or max_iou < iou: #현재까지의 최대 IoU값보다 현재 Iou값이 더크면
                max_iou = iou #최대 iou값을 현재 iou값으로 업데이트
                max_gt = gt #최대 iou를 가진 정답 상자를 현재 정답 상자로 업데이트
        return max_gt
    
def update_history(action):
        """
            Fonction qui met à jour l'historique des actions en y ajoutant la dernière effectuée
            마지막으로 수행한 작업을 추가하여 작업 이력을 업데이트하는 기능
            Entrée :
                - Dernière action effectuée
        """
        action_vector = torch.zeros(9)
        action_vector[action] = 1
        size_history_vector = len(torch.nonzero(actions_history))
        if size_history_vector < 9:
            actions_history[size_history_vector][action] = 1
        else:
            for i in range(8,0,-1):
                actions_history[i][:] = actions_history[i-1][:]
                actions_history[0][:] = action_vector[:] 
        return actions_history
    
    
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
def optimizer_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    next_states = [s for s in batch.next_state if s is not None]
    non_final_next_states = Variable(torch.cat(next_states), 
                                     volatile=True).type(Tensor)
    state_batch = Variable(torch.cat(batch.state)).type(Tensor)
    action_batch = Variable(torch.LongTensor(batch.action).view(-1,1)).type(LongTensor)
    reward_batch = Variable(torch.FloatTensor(batch.reward).view(-1,1)).type(Tensor)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE, 1).type(Tensor)) 
    next_state_values[non_final_mask.view(-1,1)] = model(non_final_next_states).max(1)[0]
    
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
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
    xmin = 0.0
    xmax = 224.0
    ymin = 0.0
    ymax = 224.0
    now = time.time()
    for key, value in  train_loader.items(): #훈련데이터로에서 이미지와 정답상자를 추출
                image, ground_truth_boxes = extract(key, train_loader)#현재 이미지와 해당 이미지의 정답 상자를 추출
                original_image = image.clone() #원본이미지 보존 위해 현재 이미지를 복사
                ground_truth = ground_truth_boxes[0] #정답 상자중 첫번째 상자를 가져옴
                all_actions = [] #현재 에피소드에서 수행한 모든 액션을 저장하는 리스틀 초기화
        
                # Initialize the environment and state
                self.actions_history = torch.ones((9,9)) #액션 히스토리 텐서를 1로 초기화 한다.
                state = self.compose_state(image) #현재 이미지를 상태로 변환 cnn이용
                original_coordinates = [xmin, xmax, ymin, ymax] #원본이미지의 경계상자 좌표를 나타내는 리스트를 초기화
                new_image = image #현재 이미지를 새이미지로 설정
                done = False
                t = 0
                actual_equivalent_coord = original_coordinates #현재 상자의 좌표를 나타내느 변수를 초기화
                new_equivalent_coord = original_coordinates # 새로운 상자의 좌표를 나타내는 변수를 초기화
                while not done: #에피소드가 종료 될때 까지 반복
                    t += 1
                    action = self.select_action(state, all_actions, ground_truth) #상태에 기반하여 액션을 선택합니다.
                    all_actions.append(action) #현재 action을 리스트에 추가
                    if action == 0:#action이 0이면 종료조건에 도달한 것으로 판단
                        next_state = None
                        new_equivalent_coord = self.calculate_position_box(all_actions) #모든 action을 기반으로 새로운 상자 좌표를 계산
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord ) #가장 가까운 정답 상자를 찾습니다 정답상자는 여러개일 수 있다 객체가 여러개 일수 있으므로
                        reward = self.compute_trigger_reward(new_equivalent_coord,  closest_gt) #종료 보상을 계산
                        done = True #에피소드 종료를 표시

                    else: #액션이 0이 아닌경우
                        self.actions_history = self.update_history(action) #히스토리 업데이트
                        new_equivalent_coord = self.calculate_position_box(all_actions) #모든 액션을 기반으로 새로운 상자 좌표를 계산
                        
                        new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                        try:
                            new_image = transform(new_image)
                        except ValueError:
                            break                        

                        next_state = self.compose_state(new_image)
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, closest_gt)
                        
                        actual_equivalent_coord = new_equivalent_coord
                    if t == 20:
                        done = True
                    self.memory.push(state, int(action), next_state, reward)

                    # Move to the next state
                    state = next_state
                    image = new_image
                    # Perform one step of the optimization (on the target network)
                    self.optimize_model()
                    
            
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if i_episode<5:
                self.EPS -= 0.18
            self.save_network()

            

    for i in range(len(image_names)):
        # the image part
        image_name = image_names[i]
        image = images[i]
        if i < len(image_names_1):
            annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc_1)
        else:
            annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc_2)            
        classes_gt_objects = get_ids_objects_from_annotation(annotation)  # 주석으로부터 객체 id들을 가져옴 
        gt_masks = generate_bounding_box_from_annotation(annotation, image.shape) # 주석으로부터 gtbox를 가져온다
         
        # the iou part
        original_shape = (image.shape[0], image.shape[1])
        region_mask = np.ones((image.shape[0], image.shape[1]))
        #choose the max bouding box
        iou = find_max_bounding_box(gt_masks, region_mask, classes_gt_objects, CLASS_OBJECT)
        
        # the initial part
        region_image = image
        size_mask = original_shape
        offset = (0, 0)
        history_vector = torch.zeros((4,6)) # 체인지 스텝까지는 수정
        state = get_state(region_image, history_vector, model_vgg) # 수정
        
        original_image = image.clone() #원본이미지 보존 위해 현재 이미지를 복사
        ground_truth = gt_masks[0] #정답 상자중 첫번째 상자를 가져옴 처음엔 전체 이미지니깐 첫번째 상자를 가져와도 상관없다
        all_actions = [] #현재 에피소드에서 수행한 모든 액션을 저장하는 리스틀 초기화
        
        # Initialize the environment and state
        actions_history = torch.ones((9,9)) #액션 히스토리 텐서를 1로 초기화 한다.
        state = get_state(image, actions_history, model_vgg, dtype=torch.cuda.FloatTensor )
        original_coordinates = [xmin, xmax, ymin, ymax] #원본이미지의 경계상자 좌표를 나타내는 리스트를 초기화
        new_image = image #현재 이미지를 새이미지로 설정
        done = False
        t = 0
        actual_equivalent_coord = original_coordinates #현재 상자의 좌표를 나타내느 변수를 초기화
        new_equivalent_coord = original_coordinates # 새로운 상자의 좌표를 나타내는 변수를 초기화
        while not done: #에피소드가 종료 될때 까지 반복
                    t += 1
                    action = select_d_action(state, all_actions, ground_truth) #상태에 기반하여 액션을 선택합니다.
                    all_actions.append(action) #현재 action을 리스트에 추가
                    if action == 0:#action이 0이면 종료조건에 도달한 것으로 판단
                        next_state = None
                        new_equivalent_coord = calculate_position_box(all_actions) #모든 action을 기반으로 새로운 상자 좌표를 계산
                        closest_gt = get_max_bdbox( gt_masks, new_equivalent_coord ) #가장 가까운 정답 상자를 찾습니다 정답상자는 여러개일 수 있다 객체가 여러개 일수 있으므로
                        reward = compute_trigger_reward(new_equivalent_coord,  closest_gt) #종료 보상을 계산
                        done = True #에피소드 종료를 표시

                    else: #액션이 0이 아닌경우
                        actions_history = update_history(action) #히스토리 업데이트
                        new_equivalent_coord = calculate_position_box(all_actions) #모든 액션을 기반으로 새로운 상자 좌표를 계산
                        
                        new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                        try:
                            new_image = transform(new_image) #trnsform 전체선언 처리
                        except ValueError:
                            break                        

                        next_state = compose_state(new_image)
                        closest_gt = get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = compute_reward(new_equivalent_coord, actual_equivalent_coord, closest_gt)
                        
                        actual_equivalent_coord = new_equivalent_coord
                    if t == 20:
                        done = True
                    self.memory.push(state, int(action), next_state, reward)

                    # Move to the next state
                    state = next_state
                    image = new_image
                    # Perform one step of the optimization (on the target network)
                    self.optimize_model()

        
        for step in range(change):
            # Select action, the author force terminal action if case actual IoU is higher than 0.5
            if iou > 0.5:
                action = 9
            else:
                action = select_d_action(state)
                
            all_actions.append(action)
            
            # Perform the action and observe new state
            if action == 9:
                next_state = None
                done = True #에피소드 종료를 표시
                reward = get_reward_trigger(iou) # 기존 코드
            
            else:
                offset, region_image, size_mask, region_mask = get_crop_image_and_mask(original_shape, offset,
                                                                   region_image, size_mask, action)
                # update history vector and get next state
                history_vector = update_history_vector(history_vector, action)
                next_state = get_state(region_image, history_vector, model_vgg)
                
                # find the max bounding box in the region image
                new_iou = find_max_bounding_box(gt_masks, region_mask, classes_gt_objects, CLASS_OBJECT)
                reward = get_reward_movement(iou, new_iou)
                iou = new_iou
                ####################################################################
                self.actions_history = self.update_history(action) #히스토리 업데이트
                new_equivalent_coord = self.calculate_position_box(all_actions) #모든 액션을 기반으로 새로운 상자 좌표를 계산
                        
                new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                try:
                    new_image = transform(new_image)
                except ValueError:
                    break                        

                next_state = self.compose_state(new_image)
                closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, closest_gt)    
            
            # Store the transition in memory
            memory.push(state, action-1, next_state, reward)
            
            # Move to the next state
            state = next_state
            
            # Perform one step of the optimization (on the target network)
            optimizer_model()
            if done:
                break
    if epsilon > 0.1:
        epsilon -= 0.1
    time_cost = time.time() - now
    print('epoch = %d, time_cost = %.4f' %(epoch, time_cost))
    
# save the whole model
Q_NETWORK_PATH = '../models/' + 'voc2012_2007_model_dog'
torch.save(model, Q_NETWORK_PATH)
print('Complete')