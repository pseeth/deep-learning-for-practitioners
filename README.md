# Deep Learning For Practitioners

## Course communication

As the course is now mostly remote, we’re gonna have to do things a bit differently. There are three main ways that we’ll be communicating: Slack, Video Calls, and Canvas (mostly just for submitting assignments).

I’ve created a Slack for the course. You should have an email in your Northwestern-associated inbox with a link to sign up.

I’ve decided the best way to do things is to have active Slack discussions + video calls about papers/code/etc. We’ll have designated times to sign on and discuss these things. Slack participation will be very important. I’ll also set up a video call that will be in parallel with the Slack discussion. This is to facilitate both poor-internet and good-internet scenarios, depending on what your work-from-home situation is right now. If your work-from-home situation isn’t great, please message me and we can discuss how to make things work. But you must join the Slack and be active on it. That’s the best way to make this class work and for everyone to get the most out of it. You’ll learn the most from good discussion more than lecture! I recommend downloading the desktop app to stay connected and not just using a browser tab.

We will have both synchronous as well as asynchronous components to this course. Synchronous components will take the form of weekly discussions at a designated time: **Wednesdays from 2-5PM CDT (Chicago time)**. You must be on time for the synchronous component! The asynchronous component will in the form of Slack discussions, offline group meetings that you set yourself, and office hours with me, by appointment.

## Paper/code presentations

A big component of this course is paper AND code presentations. We’ll be doing active paper discussions and deep dives into code every week. Each week, two groups (2 people per group) will be asked to present a paper + code-base each. What paper and what code-base you present is up to you, but I’ll provide a list of topics I think would be interesting. 

Each presentation will have three components: 

- Dissemination of supporting material such as slides, code, notebooks, papers, blogposts, etc. This will happen asynchronously and can be iterated on as much as you want. If you improve materials after your presentation, please let the class know on Slack, with links! The goal is to come up with a very clean explanation of a topic you care about, which takes iteration. 
- Presentation and discussion synchronously via Slack and video call. After the presentation, there will be Q&A, via Slack or video call. If a question comes up that you have to think about, you can answer it on Slack later. Each week of the course will have a Slack channel dedicated to it. Please show up on time for synchronous discussion.
- Scribe: Two people will be assigned to be the scribe for each week, one for each presentation. The scribe should note down any questions, the answers given, and any other notes for the presenters. The presenters should then take this into account and improve their supporting materials.

I’ll be doing the first presentation (this week), which will be focused around a few landmark papers in using deep learning for computer audition, along with a code-base I wrote myself. This first presentation will show you the level of effort I expect each week from everyone. There is a course web-page here: https://pseeth.github.io/deep-learning-for-practitioners/. Your material should be PRed (pull-requested) into that repository after you present, along with notes from the scribe. If you used slides you found online, please upload the PDF in your PR, not just the link to the slides.

## Projects

Projects are an important component of this course. We will split into groups of 3 or 4, depending how things work out. Independent projects are not encouraged as I simply will be stretched too thin to help out if everyone does that! 
The focus of this course is to build code-bases that reproduce papers you are interested in. If a paper reports N numbers, ideally your code-base should reproduce those results as much as possible If you cannot reproduce a paper, it's okay, but we need to thoroughly document why that happened! At the end of your project, you will write two documents:

- Tips and tricks for training X: a document consisting of any things you had to take care of when implementing the paper. Every deep learning project has a bunch of stuff that goes wrong while implementing it, often with bizarre effects. Every time something goes wrong, you should write down what went wrong and how you fixed it in this document. This should be a living document that grows as your project continues on.
- Reproducibility report: a document that describes your attempts at reproducing the main results from the paper. This should be a final report summarizing all of your efforts to reproduce the paper and what were the key parts to doing so or hypotheses as to why you could not do so.

To facilitate better code and consistent effort, we will be doing code reviews. Each group will review one other group’s code each week. To train the model for real on a lot of GPUs, I will use something like https://github.com/goerz/tmuxpair to give you access to a machine. This will be towards the end of the course, unless the code-reviews suggest that your code-base is ready to scale up to a large dataset early on. To test your code before that point, you can write good test cases on your own computer. To test things on a GPU, you can use Paperspace GPUs, or Colab GPUs. Once I’m convinced things work, we’ll run it for real.

The code-base I present this week will show the quality of the code I expect from everyone. Good code takes consistent effort and careful thought. The code-base I will be presenting took me around 7-8 weeks to get completely up to scratch, tested, and reproducing SOTA results. This is not easy, so rushing the code the week the last week before will not do.

Finally, some of the work you do in this course might be paper-able. If that’s the case, then I’m happy to continue on to help with getting it published! Or, you of course own your own code and can do whatever you want with it and publish on your own. Regardless, I hope this course helps with your goals!

All further communication will be done via Slack! Welcome to the course, and thanks for signing up!

Prem

## Materials

### Week one

- https://arxiv.org/pdf/1508.04306.pdf
- https://www.merl.com/publications/docs/TR2018-005.pdf

And here’s the codebase:

- https://github.com/interactiveaudiolab/nussl/tree/refactor
- https://github.com/interactiveaudiolab/nussl/blob/refactor/nussl/ml/train/loss.py#L66

We’ll also read one additional paper and discuss it both this week and next week:

- https://papers.nips.cc/paper/8787-a-step-toward-quantifying-independently-reproducible-machine-learning-research.pdf

I’ll also be handing out a “deep learning quickstart” assignment this week for everyone to do, on their own. This simple assignment will just get you into the groove of writing code, tests, and doing experiments on some simple models with MNIST, so you can train it on your laptop. I’ll send around the link very soon via Canvas.
