from process.d_trainfunction import train

datasets_paths = ["data/us8k/npy/coc_out", "data/us8k/npy/harmo_out"]

# train(datasets_paths, path2tab='tab', path2trained='bestmodel',
#       trials=2, batch_size=5, num_epochs=2, initial_lr=0.1, step_size=5, test_train=True)
#
train(datasets_paths, path2tab='tab', path2trained='bestmodel',
      trials=5, batch_size=50, num_epochs=30, initial_lr=0.1, step_size=10)

print('finish')