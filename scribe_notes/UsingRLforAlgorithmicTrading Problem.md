# Using RL for Algorithmic Trading Problem 
(Q - Question) (A - Answer)


15:30 Using RL for Algorithmic Trading Problem 


Trading over one index, testing different architectures 


Profitability == gross margin 


Moscow Exchange, specifically the RTS index (based on 50 liquid stocks)


The input data was “candlestick” data


Order data is all the information on each action taken (relevant to trading) 


15:37 Brief review of RL


The algorithm that they used A3C - Asynchronous Advantage Actor-Critic Algorithm functions kind of like a GAN where agents learn from each other. Value function 


Environment provided by Open  AI’s Gym


15:40 Deep Q-networks overview 


Q: Do you know the Q stands for in Deep learning?
A: Q - quality, stands for R


Metaphor linking experience as cache 


Use an additional network during training to generate the target-Q values 


Approaches of DQN: Value based and policy based 


Actor & Critic: like a combination of the two approaches taking the best parts of each and removing the drawbacks of each 


Q: So the actor only learns from the critic?
A: The actor does learn through the critic without ever seeing the reward. 


15:50 Advantage Actor-Critic(A2C)
Split the Q values into the advantage value A(s, a) and the value function  V(s)
Q(s, a) = V(s) + A(s, a) => A(s, a) = Q(s, a) - V(s) => A(s, a) = r + yV(s_hat) - V(s)


15:51 Asynchronous Advantage Actor-Critic A3C
Developed by Deepmind AI, the key difference is you can have multiple independent agents training on different copies of the environment in parallel. 


15:54 Annotations 
S - States ( time ) 
A - Actions which are {-1, 0, 1} 
        -1 barrow stock right away
        0 is cashing out
        1 is keeping the stock for long term
P- transition probability 
R is the reward
Gamma - discount factor 


Goal of the paper was to find the best strategy which will maximize the mathematical expectation of the reward of p^pi


 15:59 architecture overview
Input, Dense Dropout layer, LSTM, Value layer, Policy layer 


Critic helps pick the best action to be taken 


The results were that 


16:02 Code walkthrough 


Starting with the workers and agents 


The loss function helps select which probability is the most likely outcome 


One worker cannot know everything in the environment, that’s why multiple workers are used. This makes the training more efficient 


Q: 16:07 In the train section, it optimizes the advantages, but is it also storing the value of the state after each iteration?
A: They are storing the past values so that they can predict the next value. 


They used tensor flow 


(Prem) Looks like they also use gradient clipping, the reason why is because they were training RNN. To mitigate exploding networks. 


The data format was an .sdf file, but on local there were just issues loading this in. What ended up working was another portion of the code.


Q: Are you gonna deploy this yet?
A: Actually since it was based on moscow exchange, no.  
Q: Why wouldn’t it transfer from the moscow market?
A: The data has no U.S companies. If this model was trained on NASDAQ it may perform well. Also trading on historical data doesn’t work very well. Due to current conditions it would be very difficult to predict the market right now.
