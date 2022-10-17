import logging
import math
import random
import numpy as np
import time
import torch as t
import torch.nn as nn
from torch import optim
from torch.nn import utils
import matplotlib.pyplot as plt

dt = 0.1 # s, time interval
g = 0.12 # m/s^2? gravitational acceleration
boost_a = 0.18 # m/s^2? thrust constant

w = 0.25 # m, landing platform width
h = 0.06 # m, landing platform height
r_a = 20 # rad/s^2? ratational constant


class Dynamics(nn.Module):

    def __init__(self):
        super(Dynamics, self).__init__()

    @staticmethod
    def forward(state, action):

        # Apply gravity, put this as the second element in a tensor bc
        delta_state_gravity = g * dt * t.tensor([0., 1.]) # m/s, speed change due to gravitational acceleration
        # delta_state_gravity = t.tensor([0., g * dt]) # original eqn.

        # Thrust
        delta_state = boost_a * dt * t.tensor([0., -1.]) * action # original eqn.


        # New velocity after considering gravity and thrust
        state = state + delta_state + delta_state_gravity

        # Update state
        step_mat = t.tensor([[1., dt], [0., 1.]])
        state = t.matmul(step_mat, state)
        # print('State:', state, '\n Action:', action)

        return state


class Controller(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states
        dim_output: # of actions
        dim_hidden: up to you
        """
        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            # You can add more layers here
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
        # x0 = float(np.random.rand(1))*5
        # v0 = float(np.random.rand(1))

        state = [1., 0.]  # TODO: need batch of initial states
        return t.tensor(state, requires_grad=False).float()

    def error(self, state):
        return state[0]**2 + state[1]**2



class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.01)

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss

        self.optimizer.step(closure)
        return closure()

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            print('[%d] loss: %.3f' % (epoch + 1, loss))
            self.visualize()

    def visualize(self):
        data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        x = data[:, 0] # position
        y = data[:, 1] # velocity
        plt.plot(x, y)
        plt.xlabel('Position, d(t)')
        plt.ylabel('Velocity, v(t)')
        plt.show()




T = 100  # number of time steps
dim_input = 2  # state space dimensions
dim_hidden = 6  # latent dimensions
dim_output = 1  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller
s = Simulation(c, d, T)  # define simulation
o = Optimize(s)  # define optimizer
o.train(40)  # solve the optimization problem



