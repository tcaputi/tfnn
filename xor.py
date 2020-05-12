import numpy as np
import tensorflow as tf

def generate_training_data(n):
    inputs = np.random.randint(2, size=(n, 2))
    outputs_raw = []
    for inp in inputs:
        outputs_raw.append([inp[0] ^ inp[1]])

    outputs = np.array(outputs_raw, 'float32')
    return (np.asarray(inputs, dtype=np.float32), outputs)

print('creating model')
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2, input_dim=2, activation='relu'),
    tf.keras.layers.Dense(11, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

opt = tf.keras.optimizers.SGD(lr=0.15, momentum=0.5)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['binary_accuracy'])

print('training model')
(train_inputs, train_outputs) = generate_training_data(20000)
print(train_inputs)
print(train_outputs)
model.fit(train_inputs, train_outputs, epochs=1)

print('testing model')
(test_inputs, test_outputs) = generate_training_data(1000)
model.evaluate(test_inputs, test_outputs)
