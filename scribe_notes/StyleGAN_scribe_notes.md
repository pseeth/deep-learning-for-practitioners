# StyleGAN Scribe Notes

## Questions:
### StyleGAN:
1. Bonga: Is the loss function of the discriminator for both the generator and input real image?
- Yes

2. Prem: Mapping network looks familiar. What does it look like?
- Essentially it’s like FiLM (presented in week 4)

3. Yao Gu: What does AdaIN do here?
- Normalizes mean and variance of feature map

4. Ari: Can you clarify what entanglement is?
- Two features should remain as they are, separated (eye feature and hair color, you want them to remain as they are)

5. Yao Gu: Slide 10: what's Z and W in (b) (c)?
- Z is general latent space and W is intermediate latent space

6. Jiuqi: If I have a trained model and I want to change the hair from blond to black, how do I know which vector to change?
- Take all the images of people with blond and black hair respectively and see what features are changing. There’s no direct way though

7. Prem’s explanation on the video:
- Hair, background, orientation of eyes, skin tone - how these features are changing, their patterns

### StyleGAN 2:
Ari: Can you clarify what you mean by squeeze/stretch?

## Code comments/questions:
1. Prem: Closure - a nice way to structure code
2. Prem: training loop should basically look the same as DCGAN right?
- Yes

### Appendix (Authors’ comments)
They wanted to make some addition in slide 12 (more explanatory image?)
