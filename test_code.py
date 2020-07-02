test_model = NewNN(n_actions=6).to(device)
test_model.load_state_dict(torch.load("/content/GO1.pth"))
state = env.reset()
featureList=[]
qList=[]
actList=[]

for episode in range(0,20):
    obs = env.reset()
    state = get_state(obs)
    total_reward = 0.0
    for t in count():
        action = test_model(state.to(device)).max(1)[1].view(1,1)
        feature_output1 = test_model.featuremap1.transpose(1,0).cpu()
        if(t%15==0):
            #state=Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            q_v=test_model(state.to(device)).max(1)[0].view(1,1)
            qList.append(q_v)
            actList.append(action)
            featureList.append(feature_output1)
        #if render:
        #    env.render()
         #   time.sleep(0.02)

        obs, reward, done, info = env.step(action)

        total_reward += reward

        if not done:
            next_state = get_state(obs)
        else:
            next_state = None

        state = next_state

        if done:
            print("Finished Episode {} with reward {}".format(episode, total_reward))
            break

    env.close()

feaList=featureList[-1000:]
QL=qList[-1000:]

array=[t.numpy() for t in feaList]
Qarray=np.array(QL)
dataArr=np.array(array)

dataArr=dataArr.squeeze(2)

from sklearn.datasets import load_digits
from sklearn.manifold import MDS
embedding = MDS(n_components=2)
X_transformed = embedding.fit_transform(dataArr.astype(np.float64))
X_transformed.shape
X_transformed=np.transpose(X_transformed)

from mpl_toolkits import mplot3d
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 300 
fig = plt.figure()

plt.scatter(X_transformed[0],X_transformed[1],c=Qarray,s=0.5);
plt.show()

