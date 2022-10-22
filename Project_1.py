
import numpy as np
import torch as t
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

dt = 0.1 # s, time interval
g = 0.12 # gravitational acceleration
boost_a = 0.18 # thrust constant
r_a = 1 #  ratational acceleration


w = 0.25 # m, landing platform width
h = 0.06 # m, landing platform height


class Dynamics(nn.Module):

    def __init__(self):
        super(Dynamics, self).__init__()

    @staticmethod
    def forward(state, action):

        # Speed change due to gravitational acceleration
        delta_v_g = g * dt * t.tensor([0., -1., 0., 0.])

        # Speed change due to thrust
        delta_v_a = boost_a * dt * t.tensor([0., 1., 0., 0.]) * action[0]

        # Angular velocity change due to rotational thrust
        delta_omega = r_a * dt * t.tensor([0., 0., 0., -1.]) * (2*action[1]-1)

        # Apply changes to velocity and angular velocity
        state = state + delta_v_a + delta_v_g + delta_omega

        # Update state
        step_mat = t.tensor([[1., dt, 0., 0.], [0., 1., 0., 0.], [0., 0., 1., dt], [0., 0., 0., 1.]])
        state = t.matmul(step_mat, state)

        return state


class Controller(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_hidden2, dim_output):
        """
        dim_input: # of system states
        dim_output: # of actions
        dim_hidden: up to you
        """
        # dim_hidden2 = dim_hidden/2+1
        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden2),
            nn.Mish(),
            nn.Linear(dim_hidden2, dim_output),
            nn.Sigmoid()
        )

    def forward(self, state):
        action = self.network(state)
        return action



class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.action_trajectory = []
        self.state_trajectory = []

    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(T):
            action = self.controller.forward(state)
            state = self.dynamics.forward(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error(state)

    @staticmethod
    def initialize_state():
        y0 = float(np.random.rand(1))*5 # this makes y0 = [0, 5]
        v0 = float(np.random.rand(1))*-0.2 # this makes v0 = [-0.2, 0]
        phi0 = float(np.random.rand(1))*2-1 # this makes phi0 = [-1, 1]
        omega0 = float(np.random.rand(1))*2-1 # this makes omega0 = [-1, 1]

        state = [y0, v0, phi0, omega0]
        print('Initial state:', state)
        # state = [1., 0., 0.5, 0.]
        return t.tensor(state, requires_grad=False).float()

    def error(self, state):
        return state[0]**2 + state[1]**2 + state[2]**2 + state[3]**2



class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters() # can find gradient from self.parameters.grad?
        self.optimizer = optim.LBFGS(self.parameters, lr=0.01)

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward() # gradient descent step
            return loss

        self.optimizer.step(closure)
        return closure()

    def train(self, eps):
        loss = 1
        epoch = 0
        x0 = self.simulation.state
        x0 = np.array(x0)
        while loss >= eps and epoch < 200: # should solve in less than 200 steps
            epoch += 1
            loss = self.step()
            print('[%d] loss: %.6f' % (epoch + 1, loss))
        if loss <= eps: # only plot if it was a successful run
            self.visualize(epoch, x0) # plot once it is converged (last gradient descent step)

    def visualize(self, i, x0):
        data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        y = data[:, 0] # position
        v = data[:, 1] # velocity
        phi = data[:, 2] # angular position
        omega = data[:, 3] # angular velocity

        plt.figure()
        plt.subplot(211)
        plt.plot(y, v)
        plt.xlabel('Position, d(t)')
        plt.ylabel('Velocity, v(t)')
        # plt.title('gradient Descent Iteration: '+str(i)+'\n'+'x0='+str(x0))
        plt.subplot(212)
        plt.plot(phi, omega)
        plt.xlabel('Angular Position, \u03A6(t)') # printing greek letters: https://pythonforundergradengineers.com/unicode-characters-in-python.html
        plt.ylabel('Angular Velocity, \u03C9(t)')
        plt.subplots_adjust(hspace=0.3)
        plt.savefig('.\\Results_Figures\\GD_step='+str(i)+'_x0='+str(x0)+'.jpg', dpi=300, bbox_inches='tight')
        plt.show()



eps = 1e-6
T = 100  # number of time steps
dim_input = 4  # state space dimensions
dim_hidden = 5  # latent dimensions
dim_hidden2 = 3  # latent dimensions 2
dim_output = 2  # action space dimensions

d = Dynamics()  # define dynamics
for i in range(100): # test 100 different initial states
    c = Controller(dim_input, dim_hidden, dim_hidden2, dim_output)  # define controller, need to call this to initialize a new initial state
    s = Simulation(c, d, T)  # define simulation
    o = Optimize(s)  # define optimizer
    o.train(eps)  # solve the optimization problem



