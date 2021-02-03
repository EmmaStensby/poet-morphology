# Poet with morphology evolution

This repository contains the source code for our paper *Co-optimising Robot Morphology and Controller in a Simulated Open-ended Environment*.

**Abstract.** Designing robots by hand can be costly and time consuming, especially if the robots have to be created with novel materials, or be robust to internal or external changes. In order to create robots automatically, without the need for human intervention, it is necessary to optimise both the behaviour and the body design of the robot. However, when co-optimising the morphology and controller of a locomoting agent the morphology tends to converge prematurely, reaching a local optimum. Approaches such as explicit protection of morphological innovation have been used to reduce this problem, but it might also be possible to increase exploration of morphologies using a more indirect approach.

We explore how changing the environment, where the agent locomotes, affects the convergence of morphologies. The agents' morphologies and controllers are co-optimised, while the environments the agents locomote in are evolved open-endedly with the Paired Open-Ended Trailblazer (POET). We compare the diversity, fitness and robustness of agents evolving in environments generated by POET to agents evolved in handcrafted curricula of environments.

We show that agents evolving in open-endedly evolving environments exhibit larger morphological diversity in their population than agents evolving in hand crafted curricula of environments. POET proved capable of creating a curriculum of environments which encouraged both diversity and quality in the population. This suggests that POET can be a promising approach to reduce premature convergence in co-optimisation of morphology and controllers.

<img src="https://github.com/EmmaStensby/poet-morphology/blob/main/readme_images/agent_example.gif" alt="gif1" width="380" align="left" style="margin: 10px"/> <img src="https://github.com/EmmaStensby/poet-morphology/blob/main/readme_images/agent_example_2.gif" alt="gif2" width="380" align="right" style="margin: 10px"/>
