This is an assigment from the course Deep Reinforcement Learning (Ben Gurion University of the Negev).

Implementation of **value iteration algorithm** and **policy iteration algorithm** to find the optimal policy for a **Markov Decision Process**.  

In this example we have a 4X4 grid world with 5 terminal states.  
4 are holes and one is the goal state. In every state we have a 0.8 probability that the agent will take the action required and 0.1 probability that the agent will go left or right.  
This example is illustrated by a penguin looking for the optimal policy that will lead it to the fish (goal state) while trying to avoid falling into the holes. The icy floor is slippery thus the penguin may slip and wonder to the sides. In this examples the boundaries are reflective. 
