## AutoGAN Scribe Notes

1. GANs have a generator and a discrimator, one acting adversarially towards the other: the generator's goal is to maximize classification error while the discriminator's goal is to minimize it.

2. Progressive Growing GANs follow a recipe based on increasingly higher resolutions for the generated images, presumably increasing the difficulty in discriminating between false and true images.

3. Inception score accounts for both the quality and the diversity of the generated images.

4. Current problems refer to excessive variance/sensitivity of results w.r.t the architecture and parameter space used.

6. Neural architecture search aims at guiding the GAN over the decision space, with higher-level parameters in the sense that they represent design decisions for composing the neural architecture. Controller RNN is a network that performs neural architecture search such that we can we frame its improvement as a RL task. It does multiple models at once and gathers rewards for this batch of models.

7. We then specify the search space of AutoGAN -- in essence, different design choices for the architecture -- and choose inception score over FID score because the former is less time-consuming than the latter. Furthermore, we search the space progressively and similarly to the idea of Progressive Growing GANs, but with the RNN architecture.

8. The experiments comprise two datasets, CIFAR-10 and STL-10 (the latter to examine the transferability of AutoGAN). Results for CIFAR-10 are superior to most of prior work, but ambiguous when compared to Progressive GAN. Results for STL-10 are comparable to prior works', therefore AutoGAN is not overfitting to training data and its transferability/robustness is presumably comparable to other GAN architectures.
---
### Key takeaways:

9. So, in conclusion, we have AutoGAN as a means for searching the many design decisions one can make when setting up a GAN, leading to results that are comparable to prior works in terms of performance and robustness, despite being less performatic when compared to "fine-tuned" Progressive GANs. It's unclear to me how much time it takes and it saves by automating this aspect of the experimentation (update with Prem's comment: search takes 43 hours for STL-10!).
