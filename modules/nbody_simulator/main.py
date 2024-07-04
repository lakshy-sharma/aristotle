import copy
from math import ceil
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

class ResearchController:

    def __init__(self, StateInit=None, masses=[], G=6.67408e-11) -> None:

        # Init state and masses.
        self.StateInit = StateInit
        self.masses = masses
        self.G = G
        self.history = None
        self.energies = None
        return None

    def get_energy(self, State):
        """
        This function calculates the kinetic and potential enrgy of a given state.

        Inputs

        Returns
        """
        number_of_bodies, dimensions = State.shape
        dimensions = dimensions // 2
        positions = State[:,:dimensions]
        velocities = State[:,dimensions:]

        KE = 0
        for i in range(number_of_bodies):
            KE += 0.5 * self.masses[i] * (np.linalg.norm(velocities[i]) ** 2)

        PE = 0
        for i, j in self.pairs:
            r = np.linalg.norm(positions[j] - positions[i])
            PE -= (self.masses[i] * self.masses[j])/ r
        PE *= self.G

        return KE, PE

    def get_state_derivative(self, State):
        """
        This function determines the state derivative for a given state.
        """
        number_of_bodies, dimensions = State.shape
        dimensions = dimensions // 2
        positions = State[:, :dimensions]
        velocities = State[:,dimensions:]

        Statedot = np.zeros_like(State)
        Statedot[:, :dimensions] = velocities

        for i, j in self.pairs:
            r1, r2 = positions[i], positions[j]
            relative_position = r2- r1
            distance = np.linalg.norm(relative_position)
            F = self.G * self.masses[i] * self.masses[j] * relative_position / (distance ** 3)
            a1 = F / self.masses[i]
            a2 = F / self.masses[j]

            # Apply acceleration to i and j objects.
            Statedot[i, dimensions:] += i
            Statedot[j, dimensions:] += j

        return Statedot

    def rk4_integrator(self,State, dt, evaluate):
        """
        This function performs integration of state and calculates the state after 1 timestep.
        """
        k1 = evaluate(State)
        k2 = evaluate(State + 0.5*k1*dt)
        k3 = evaluate(State + 0.5*k2*dt)
        k4 = evaluate(State + k3*dt)
        
        # Update X
        Stateprime = (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        return State + Stateprime * dt

    def start(self, T, dt) -> np.ndarray:
        """
        Inputs:
            T : Total runtime of the simluation
            dt: Timestep for integration.

        Returns:
            history: A Matrix of history of states.
        """
        assert self.StateInit is not None
        assert self.masses is not None

        iterations = ceil(T/ dt)
        number_of_bodies, dimensions = self.StateInit.shape
        self.history = np.zeros((iterations + 1, number_of_bodies, dimensions))
        self.history[0]

        self.pairs = list(combinations(range(number_of_bodies), 2))
        self.energies = np.zeros((iterations+1, 3))
        KE, PE = self.get_energy(self.StateInit)

        self.energies[0] = np.array([KE, PE, KE+PE])

        State = copy.deepcopy(self.StateInit)
        for i in range(iterations):
            State = self.rk4_integrator(State, dt, self.get_state_derivative)
            self.history[i+1] = State
            KE,PE = self.get_energy(State)
            self.energies[i+1] = np.array([KE, PE, KE+PE])

        return self.history