### How to deal with tensorflow.data.Dataset for huge and multihead models
# My Continues Distance Dependent Positional Encoding (CDDPE) technique
def getPositionEncoding(max_length, relative_positions, dim=38, n=500):
    relative_positions = np.squeeze(relative_positions, axis=0)
    j = np.where(relative_positions >= 900, relative_positions - 1000, relative_positions)
    j = j.astype(np.float32)
    b = np.where(relative_positions >= 900, 2.1, 0.0).astype(np.float32)
    i = np.arange(dim // 2, dtype=np.float32)
    denominator = n ** (2 * i / dim)
    P_k = np.stack([
        np.sin(j[:, None] / denominator) + b[:, None],
        np.cos(j[:, None] / denominator) + b[:, None]
    ], axis=-1).reshape(-1, dim)
    paddings = [[0, max_length - P_k.shape[0]], [0, 0]]
    P = np.pad(P_k, paddings, "constant")
    return P


def path_gen(file_paths):
    for file in file_paths:
        yield file
def load_process(file):
    # Convert the tensor to a Python string
    file_path = file.numpy().decode("utf-8")
    # Load data from the file
    dicti = np.load(file_path)
    # Process the loaded data
    channel, embed, pos = dicti['channel'], dicti['embed'], dicti['Pos']
    channel, embed, pos = tf.constant(channel, dtype=tf.float32),tf.constant(embed, dtype=tf.float32),tf.constant(pos, dtype=tf.float32)
    channel = tf.squeeze(channel, axis=0)
    embed = tf.squeeze(embed, axis=0)
    pos = getPositionEncoding(210, pos, dim=38, n=500)
    label = tf.squeeze(tf.constant(dicti["BA"], dtype=tf.float32), axis=0)
    
    return channel, embed, pos, label

def load_process_wrapper(file):
    return tf.py_function(load_process, [file], [tf.float32, tf.float32, tf.float32, tf.float32])

def load_label(file):
    dicti = np.load(file)
    label = tf.squeeze(dicti["BA"], axis=0)
    return label
def extract_components(channel, embed, pos, label):
    return {'input_1': channel, 'input_2': embed, 'input_3': pos}, label

def dataset_generator(file_path_list, batch_size, epochs, buffer_size, shuffle=True):
    trds = tf.data.Dataset.from_tensor_slices(file_path_list)
    if shuffle:
        trds = trds.shuffle(buffer_size)
    trds = trds.map(load_process_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    trds = trds.map(extract_components, num_parallel_calls=tf.data.AUTOTUNE)
    trds = trds.batch(bs)
    trds = trds.prefetch(tf.data.AUTOTUNE)
    trds = trds.repeat(epochs)
    return trds
