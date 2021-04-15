def get_model(hp):
    input_shape = trainX[0].shape
    inputs = tf.keras.Input(shape=input_shape)

    print(inputs)
    
    x = inputs
    
    for i in range(hp.Int('conv_layer', 1, 6, default=3)):
        kernel = hp.Int('kernel_size_' + str(i), 3, 50)
        filters = hp.Int('num_filter_' + str(i), 1, 32)
        
        x = tf.keras.layers.Conv3D(filters, kernel, activation='relu', kernel_initializer='he_uniform')(x)
    
        pooling = hp.Choice('pooling_' + str(i), values=[True, False])
        
        if pooling:
            pool = hp.Int('pooling_size_' + str(i), 2, 10)
            x = tf.keras.layers.MaxPooling3D(pool_size=(pool, pool, pool))(x)
    
    
    x = tf.keras.layers.Flatten()(x)

    for i in range(hp.Int('hidden_layers', 1, 6, default=3)):
        size = hp.Int('hidden_size_' + str(i), 10, 700, step=40)
        reg = hp.Float('hidden_reg_' + str(i), 0,
                       0.06, step=0.01, default=0.02)
        dropout = hp.Float('hidden_dropout_' + str(i),
                           0, 0.5, step=0.1, default=0.2)

        x = tf.keras.layers.Dense(size, activation="relu",
                                  kernel_regularizer=regularizers.l2(reg))(x)
        x = tf.keras.layers.Dropout(dropout)(x)

        norm = hp.Choice('hidden_batch_norm_' + str(i), values=[True, False])

        if norm:
            x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(1, kernel_regularizer='l2')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-6, 1e-4, sampling='log')),
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.MeanSquaredError()])

    return model