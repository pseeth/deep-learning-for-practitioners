import matplotlib.pyplot as plt

from bn.train_linear import train, ModelType

l1_mean, l1_std, l2_mean, l2_std, loss_arr, model_type = train(model_type=ModelType.LINEAR, num_epochs=10)
l1_mean_bn, l1_std_bn, l2_mean_bn, l2_std_bn, loss_arr_bn, model_type_bn = train(model_type=ModelType.LINEAR_BN, num_epochs=10)
_, _, _, _, _, _ = train(model_type=ModelType.LINEAR_BN_NOISE, num_epochs=10)

"""
Move plots to another file if they get in the way
"""
# # plot layer 1
# plt.plot(l1_mean, 'g', label='%s layer 1 input mean' % model_type)
# plt.plot(l1_mean_bn, 'b', label='%s layer 1 input mean' % model_type_bn)
# plt.title('Layer 1 Mean: %s vs. %s' % (model_type, model_type_bn))
# plt.ylabel('Mean Value Before Activation')
# plt.xlabel('Iteration')
# plt.legend(loc='upper left')
# plt.savefig('imgs/l1_mean.png')
# plt.close()
#
# # plot layer 2
# plt.plot(l2_mean, 'g', label='%s layer 2 input mean' % model_type)
# plt.plot(l2_mean_bn, 'b', label='%s layer 2 input mean' % model_type_bn)
# plt.title('Layer 2 Mean: %s vs. %s' % (model_type, model_type_bn))
# plt.ylabel('Mean Value Before Activation')
# plt.xlabel('Iteration')
# plt.legend(loc='upper left')
# plt.savefig('imgs/l2_mean.png')
# plt.close()
#
# # plot model loss
# plt.plot(loss_arr, 'b', label='%s loss' % model_type)
# plt.plot(loss_arr_bn, 'g', label='%s loss' % model_type_bn)
# plt.title('Loss: %s vs. %s' % (model_type, model_type_bn))
# plt.legend(loc='upper right')
# plt.savefig('imgs/loss.png')
# plt.close()
