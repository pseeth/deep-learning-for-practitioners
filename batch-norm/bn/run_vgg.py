from bn.train_vgg import train_vgg, ModelType


# train_vgg(model_type=ModelType.VGG11, batch_norm=False)
# train_vgg(model_type=ModelType.VGG11, batch_norm=True)
train_vgg(model_type=ModelType.VGG11, batch_norm=True, noise=True)