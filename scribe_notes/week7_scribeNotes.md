# Scribe Notes for Learning to Learn How to Learn: Self-Adaptive Visual Navigation using Meta-Learning
## Presentation:

<ol>

<li>Learning to Learn: Gradient Based Meta-Learning slide - Theta minus that little thing is the update step. This is written in a weird way, but that's what it's saying.
<ul><li>Jiuqi: I think the key idea to understand this is that the model knows the model that is trained in the training time will be adapted during the training time so the MAML will be adapted during the training. They will learn as such and do updates.</li></ul>
</li>

<li>Prem: I got a little lost about the ineraction loss
<ul><li>The interaction loss is the neural network that is parameterized by phi. With the theta parameters we use they are updated by interaction loss.</li></ul>
</li>


<li>Hand Crafted Interaction Objectives slide - What's g on the slide?
<ul><li>A regular objective function. They have not mentioned the details and we could not find it in the code as well. g calculates the similarity between the states. It would try to find Sj's that minimize this. It checks the differences between the states pixel by pixel. It's like a feature map of states. </li></ul>
</li>


<li>Self Adaptation Video Examples - Victor: Reminded me of this AI2 competition:  https://ai2thor.allenai.org/</li>

<li>Self Adaptation Video Examples - Ari: Me and Zane worked on a similar project last quarter. Ours would also get stuck on things that looked like other things. We would tell it to look for the tv and it would get stuck on a window.</li>


</ol>

## Code:
<ol>

<li>Bonga: is it trained on the entire image for the object? Or is there an embedding for every word? </li>
<ul><li>They have objects you can choose from that are all GLOVE embeddings</li></ul>

<li>Victor: This ablation study part could be very interesting: imagine if ResNet worked for RoboTHOR (the real apartment) better than for iTHOR (the virtual environment)?</li>

<li>Prem: Even just swapping the vision network for other vision networks could be interesting</li>


</ol>