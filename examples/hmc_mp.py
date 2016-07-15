import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Data generation
# first class
m1 =  [-2.5, -2.5]
cov1 = [[2, 0], [0, 2]]
x1, y1 = np.random.multivariate_normal(m1,cov1,10).T
x1.clip(-10, 0)
y1.clip(-10, 0)


# second class
m2 =  [2.5, 2.5]
cov2 = [[2, 0], [0, 2]]
x2, y2 = np.random.multivariate_normal(m2,cov2,10).T
x2.clip(0, 10)
y2.clip(0, 10)


#plot toyclass
plt.plot(x1, y1, 'rx')
plt.plot(x2,y2,'bo')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()



# prepare data
num_labels = 2
input_size = 2
hidden_neuron_num = 10

x = np.concatenate((x1,x2))
y = np.concatenate((y1,y2))
labels  =  np.concatenate((np.zeros(x1.shape[0]), np.ones(x2.shape[0])))

dataset_all = np.stack((x, y)).T.astype(np.float32)  
labels_all = (np.arange(num_labels) == labels[:, None]).astype(np.float32)


data_num = dataset_all.shape[0]
    
    
order = np.random.permutation(data_num)
train_input = dataset_all[order,:]
train_output = labels_all[order,:]
    
x_test = np.linspace(-10,10,200)
y_test = np.linspace(-10,10,200)
xv_test, yv_test = np.meshgrid(x_test,y_test)


test_input = np.stack((xv_test.ravel(), yv_test.ravel())).T.astype(np.float32)
test_output = np.zeros(test_input.shape[0])



# HMC multiple particles
from mjhmc.misc.distributions import TensorflowDistribution
from mjhmc.samplers.markov_jump_hmc import ControlHMC



def energy_grad(state):
    """
	 Define energy and gradient function for toy classfication problem: 
	 Prior: Gaussian prior 
	 Likihood: Cross_entropy error function 
	 Gradient: use tensorflow built-in function for gradient calculation  

	 Args: 
		state to sample from

	 Returns:
	 	list of loss ops, and list of grad ops
    """

    tf_train_dataset = tf.constant(train_input)
    tf_train_labels = tf.constant(train_output)
    tf_test_dataset = tf.constant(test_input)
    
    loss_list = []
    grad_list = []
    state_list = tf.split(1, init.shape[1], state)

    for i in range(len(state_list)):
        # loop body
        
        state_one = tf.squeeze(state_list[i])
        w_h = tf.reshape(state_one[0:input_size*hidden_neuron_num], [input_size, hidden_neuron_num])
        b_h = state_one[20:20+hidden_neuron_num]
        w_o = tf.reshape(state_one[30:30+hidden_neuron_num*num_labels], [hidden_neuron_num, num_labels])    
        b_o = state_one[50: 50+num_labels] 
        #MLP model
        h = tf.nn.relu(tf.matmul(tf_train_dataset, w_h) + b_h)
        logits = tf.matmul(h, w_o) + b_o
        loss = (tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + \
                        0.05*tf.add_n([tf.nn.l2_loss(state)]))/data_num  
        
        grad = tf.gradients(loss, state_one, gate_gradients=[-1.0, +1,0])
        loss_list.append(loss)
        grad_list.append(grad[0])
        
    for gradient_check in grad_list:
        checked_sg = tf.check_numerics(gradient_check, 'nan in gradient of {}'.format(gradient_check.op.name))
    return loss_list, grad_list


n_dims = 52
n_particles = 2
# you're going to want to use a better initalization later
# this is shitty and will require a long burn in time
init = 0.01*np.random.randn(n_dims,n_particles)
model_distr = TensorflowDistribution(energy_grad, init, name='model_distr')

hmc = ControlHMC(distribution=model_distr)  
samples = hmc.sample(1000)
samples