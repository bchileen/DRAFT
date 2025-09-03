# model.py

import tensorflow as tf  # Keep this
# Remove or comment out these lines:
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation

from config import (
    INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH,
    LSTM_UNITS_1, LSTM_UNITS_2, LSTM_DROPOUT_1, LSTM_DROPOUT_2,
    LSTM_ACTIVATION, DENSE_ACTIVATION, FINAL_ACTIVATION,
    OPTIMIZER_LR, N_FEATURES_TO_KEEP
)

def build_lstm_model(n_features: int) -> tf.keras.models.Sequential: # Use tf.keras... here
    """Builds the LSTM model architecture."""
    print("Building LSTM model...")
    if n_features <= 0:
        raise ValueError("Number of features must be positive.")

    opt = tf.keras.optimizers.Adam(learning_rate=OPTIMIZER_LR) # Use tf.keras...

    model = tf.keras.models.Sequential(name=f"LSTM_{INPUT_SEQUENCE_LENGTH}in_{OUTPUT_SEQUENCE_LENGTH}out") # Use tf.keras...
    model.add(tf.keras.layers.LSTM( # Use tf.keras...
        LSTM_UNITS_1,
        activation=LSTM_ACTIVATION,
        return_sequences=True, # Return sequences for the next LSTM layer
        input_shape=(INPUT_SEQUENCE_LENGTH, n_features),
        name='lstm_1'
    ))
    model.add(tf.keras.layers.Dropout(LSTM_DROPOUT_1, name='dropout_1')) # Use tf.keras...

    # Add second LSTM layer (optional, based on config/analysis)
    if LSTM_UNITS_2 > 0:
        model.add(tf.keras.layers.LSTM( # Use tf.keras...
            LSTM_UNITS_2,
            activation=LSTM_ACTIVATION,
            return_sequences=False, # Only return the last output for the Dense layer
            name='lstm_2'
        ))
        model.add(tf.keras.layers.Dropout(LSTM_DROPOUT_2, name='dropout_2')) # Use tf.keras...
    # Else: If only one LSTM layer, ensure its return_sequences=False

    # Output layer
    model.add(tf.keras.layers.Dense(OUTPUT_SEQUENCE_LENGTH, activation=DENSE_ACTIVATION, name='dense_output')) # Use tf.keras...
    # Some notebooks used a final linear activation - often relu is sufficient here,
    # but adding linear if it was specifically intended.
    if FINAL_ACTIVATION == 'linear':
         model.add(tf.keras.layers.Activation('linear', name='activation_linear')) # Use tf.keras...

    model.compile(loss='mse', optimizer=opt, metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]) # tf.keras... already here

    print(model.summary())
    return model

# # Example Usage (optional)
# if __name__ == "__main__":
#     n_feats_example = N_FEATURES_TO_KEEP + 1 # Selected features + target variable
#     lstm_model = build_lstm_model(n_feats_example)
#     print("\nModel built successfully.")